"""Training config, reproducibility, package contents and the MLflow registry flow."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from creditiq_core.models import package as pkg
from creditiq_trainer import cli, pipeline
from creditiq_trainer.config import ConfigError, TrainingFileConfig, load_config
from creditiq_trainer.package_writer import write_package

ROOT = Path(__file__).resolve().parents[3]
CONFIGS = ROOT / "services" / "trainer" / "configs"
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")


def _small(**update: object) -> TrainingFileConfig:
    cfg = load_config(CONFIGS / "synthetic_ci.yaml")
    cfg.dataset.path = str(ROOT / "data" / "sample_credit_data.csv")
    return cfg.model_copy(update={"sample_size": 3000, **update})


# ---------------------------------------------------------------- config validation
@pytest.mark.parametrize("name", ["synthetic.yaml", "synthetic_ci.yaml", "home_credit.yaml", "home_credit_enriched.yaml"])
def test_bundled_configs_validate(name):
    cfg = load_config(CONFIGS / name)
    assert cfg.target.default_values and cfg.gbm_backend == "lightgbm"


def _write(tmp_path: Path, data: dict) -> Path:
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump(data))
    return p


def test_target_mapping_is_required(tmp_path):
    base = {"name": "x", "dataset": {"path": "d.csv"}, "target": {"column": "TARGET"}}
    with pytest.raises(ConfigError, match="default_values"):
        load_config(_write(tmp_path, base))
    with pytest.raises(ConfigError, match="default_values"):
        load_config(_write(tmp_path, {**base, "target": {"column": "TARGET", "default_values": []}}))


def test_invalid_configs_are_rejected_readably(tmp_path):
    ok = {"name": "x", "dataset": {"path": "d.csv"}, "target": {"column": "T", "default_values": [1]}}
    with pytest.raises(ConfigError, match="unknown_key"):
        load_config(_write(tmp_path, {**ok, "unknown_key": 1}))
    with pytest.raises(ConfigError, match="chronological"):
        load_config(_write(tmp_path, {**ok, "split": {"strategy": "chronological"}}))
    with pytest.raises(ConfigError, match="models"):
        load_config(_write(tmp_path, {**ok, "models": ["xgb_magic"]}))
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "missing.yaml")
    assert load_config(_write(tmp_path, ok)).config_hash() == load_config(_write(tmp_path, ok)).config_hash()


# ---------------------------------------------------------------- pipeline, reproducibility, package
def test_prepare_keeps_sensitive_and_ids_out_of_the_model():
    data = pipeline.prepare(_small())
    for col in ("CODE_GENDER", "DAYS_BIRTH", "NAME_FAMILY_STATUS", "SK_ID_CURR", "TARGET", "APPLICATION_DATE"):
        assert col not in data.predictors, col
    assert {"CODE_GENDER", "DAYS_BIRTH", "NAME_FAMILY_STATUS"} <= set(data.sensitive)
    assert data.info.is_synthetic and "Data Quality Report" in data.quality_report


def test_same_seed_gives_identical_runs():
    cfg = _small()
    data = pipeline.prepare(cfg)
    a, b = pipeline.train(cfg, data), pipeline.train(cfg, pipeline.prepare(cfg))
    assert a.selected == b.selected and a.thresholds == b.thresholds
    for key in a.models:
        np.testing.assert_array_equal(a.pd_split("test", key), b.pd_split("test", key))
        assert a.metrics[key]["test"] == b.metrics[key]["test"]
    c = pipeline.train(cfg.model_copy(update={"seed": 7}), pipeline.prepare(cfg.model_copy(update={"seed": 7})))
    assert not np.array_equal(c.idx("test"), a.idx("test"))  # a different seed really changes the split


def test_package_has_no_training_rows(tmp_path):
    cfg = _small()
    data = pipeline.prepare(cfg)
    result = pipeline.train(cfg, data)
    out = write_package(cfg, data, result, tmp_path / "pkg")
    for name in pkg.PACKAGE_FILES:
        assert (out / name).exists(), name
    art = pkg.artifact_from_bytes((out / pkg.ARTIFACT_FILE).read_bytes())
    assert "POST_LOAN_COLLECTION_CALLS" not in art.input_columns  # removed by the training-split leakage guard
    snap = pd.read_parquet(out / pkg.SNAPSHOT_FILE)
    assert set(snap["split"]) == {"validation", "test"}
    assert len(snap) == len(result.idx("validation")) + len(result.idx("test"))
    assert "SK_ID_CURR" not in snap.columns
    report = json.loads((out / pkg.EVALUATION_FILE).read_text())
    sel = report["models"][report["selected"]]
    assert sel["threshold_sweep_validation"] and set(sel["splits"]) == {"validation", "test"}
    meta = json.loads((out / pkg.METADATA_FILE).read_text())
    assert meta["lineage"]["creditiq.is_synthetic"] == "true" and meta["lineage"]["creditiq.config_hash"]
    assert "SYNTHETIC" in (out / pkg.MODEL_CARD_FILE).read_text()


# ---------------------------------------------------------------- MLflow: train → register → promote → rollback → export
def test_mlflow_registry_flow(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg_path = tmp_path / "ci.yaml"
    raw = yaml.safe_load((CONFIGS / "synthetic_ci.yaml").read_text())
    raw["dataset"]["path"] = str(ROOT / "data" / "sample_credit_data.csv")
    raw["sample_size"] = 3000
    cfg_path.write_text(yaml.safe_dump(raw))
    uri = f"sqlite:///{(tmp_path / 'mlruns' / 'mlflow.db').as_posix()}"
    common = ["--tracking-uri", uri, "--experiment", "test"]
    assert cli.main(["train", "--config", str(cfg_path), *common]) == 0
    assert cli.main(["train", "--config", str(cfg_path), *common, "--alias", "production"]) == 0
    from creditiq_trainer import tracking

    tracking.configure(uri, "test")
    assert tracking.alias_version("creditiq-pd", "staging") == "1"
    assert tracking.alias_version("creditiq-pd", "production") == "2"
    assert tracking.promote("creditiq-pd", "1")["replaced"] == "2"
    assert tracking.alias_version("creditiq-pd", "previous") == "2"
    assert tracking.rollback("creditiq-pd")["version"] == "2"
    versions = {v["version"]: v for v in tracking.list_versions("creditiq-pd")}
    assert versions["2"]["aliases"] == ["production"] and versions["2"]["tags"]["creditiq.is_synthetic"] == "true"
    assert cli.main(["export", *common, "--ref", "all", "--out", str(tmp_path / "reg")]) == 0
    reg = json.loads((tmp_path / "reg" / "creditiq-pd" / "registry.json").read_text())
    assert reg["aliases"]["production"] == "2" and set(reg["versions"]) == {"1", "2"}
    pkg_dir = tmp_path / "reg" / "creditiq-pd" / "versions" / "2"
    art = pkg.artifact_from_bytes((pkg_dir / pkg.ARTIFACT_FILE).read_bytes())
    assert 0 <= float(art.predict_pd({c: None for c in art.input_columns})[0]) <= 1
