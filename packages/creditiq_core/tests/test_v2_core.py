"""Tests for what v2 changed or added in the core: pure formatting, the model artifact and evaluation
snapshot, explicit boosting backends, structured drift alerts, NaN-safe JSON and the model card."""
from __future__ import annotations

import importlib.util
import json
import pickle

import numpy as np
import pandas as pd
import pytest

from creditiq_core import formatting as fmt
from creditiq_core import monitoring as mon
from creditiq_core.data import synthetic as synth
from creditiq_core.jsonsafe import to_json
from creditiq_core.models import estimators as est
from creditiq_core.models import training as mdl
from creditiq_core.models.artifact import ModelArtifact
from creditiq_core.provenance import Provenance
from creditiq_core.reporting.model_card import model_card_markdown


# ---------------------------------------------------------------- formatting (v1 money test, without Streamlit)
@pytest.mark.parametrize("cur, expected", [("", "78.84M"), ("₹", "₹78.84M"), ("$", "$78.84M"), ("INR", "INR 78.84M"),
                                           ("Rs.", "Rs. 78.84M")])
def test_money_formatting_with_currency(cur, expected):
    assert fmt.money(78_838_318.7, currency=cur) == expected
    assert fmt.money(-1234.4, compact=False, currency=cur) == "−" + expected.replace("78.84M", "1,234")
    assert fmt.amount_label("EL", cur) == (f"EL ({cur})" if cur else "EL (currency units)")


def test_formatting_handles_missing_values():
    assert fmt.money(None) == fmt.pct(float("nan")) == "n/a"
    assert fmt.val(float("nan")) == "missing" and fmt.val(1234.5) == "1,234" and fmt.val("x") == "x"


# ---------------------------------------------------------------- model artifact (replaces the v1 joblib bundle test)
def test_artifact_round_trips_and_holds_no_data_rows(experiment):
    art = experiment.to_artifact()
    restored = pickle.loads(pickle.dumps(art))
    assert isinstance(restored, ModelArtifact) and restored.format_version == 2
    X = experiment.X_split("test").iloc[:20]
    np.testing.assert_allclose(restored.predict_pd(X), art.predict_pd(X))
    assert restored.threshold == experiment.thresholds[experiment.selected]
    assert restored.threshold_rule == experiment.config.threshold_rule and restored.threshold_source == "validation"
    # No row-level data of any kind: no frames, no label or prediction arrays beyond the per-feature mean vector.
    for name, value in vars(restored).items():
        assert not isinstance(value, pd.DataFrame | pd.Series), name
    assert not hasattr(restored, "eval_val") and not hasattr(restored, "eval_test")
    assert restored.transformed_mean is not None and restored.transformed_mean.ndim == 1
    assert "score" in restored.reference and "features" in restored.reference


def test_sensitive_attributes_are_not_model_inputs(experiment):
    art = experiment.to_artifact()
    for col in ("CODE_GENDER", "DAYS_BIRTH", "NAME_FAMILY_STATUS"):
        assert col not in art.input_schema and col not in art.input_columns
    assert set(art.input_schema) == set(art.input_columns)


def test_evaluation_snapshot_has_held_out_rows_only(experiment):
    snap = experiment.evaluation_snapshot()
    assert set(snap.frame["split"]) == {"validation", "test"}
    assert len(snap.split("test")) == len(experiment.idx("test"))
    assert len(snap.split("validation")) == len(experiment.idx("validation"))
    np.testing.assert_allclose(snap.pd("test"), experiment.pd_split("test"))
    np.testing.assert_array_equal(snap.y("test"), experiment.y_split("test"))
    assert "SK_ID_CURR" not in snap.frame.columns  # identifiers are dropped
    assert {"CODE_GENDER", "DAYS_BIRTH"} <= set(snap.frame.columns)  # kept for the fairness audit only
    assert snap.exposure_col == "AMT_CREDIT" and "AMT_CREDIT" in snap.frame.columns
    with pytest.raises(ValueError, match="only"):
        snap.split("train")


def test_caller_supplied_experiment_id(synthetic_df, mapping):
    cfg = mdl.TrainingConfig(model_keys=["logreg"], sample_size=None, seed=3)
    r = mdl.run_experiment(synthetic_df, {"name": "t"}, mapping, ["EXT_SOURCE_2", "AMT_CREDIT"], cfg,
                           experiment_id="run-abc123")
    assert r.experiment_id == "run-abc123" and r.metadata()["experiment_id"] == "run-abc123"


# ---------------------------------------------------------------- explicit boosting backend
def test_named_backend_that_is_missing_fails_loudly(monkeypatch):
    real = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec", lambda name, *a: None if name == "lightgbm" else real(name, *a))
    with pytest.raises(ValueError, match="not installed"):
        est.resolve_backend("lightgbm")
    with pytest.raises(ValueError, match="Unknown"):
        est.resolve_backend("catboost")
    assert est.resolve_backend("sklearn_hgb") == "sklearn_hgb"


def test_explicit_sklearn_backend_trains(synthetic_df, mapping):
    cfg = mdl.TrainingConfig(model_keys=["gbm"], sample_size=2_000, gbm_backend="sklearn_hgb",
                             params={"gbm": {"max_iter": 40}})
    r = mdl.run_experiment(synthetic_df, {"name": "t"}, mapping, ["EXT_SOURCE_2", "AMT_CREDIT", "AMT_ANNUITY"], cfg)
    assert r.models["gbm"].backend == "sklearn_hgb"
    assert type(r.models["gbm"].estimator).__name__ == "HistGradientBoostingClassifier"


# ---------------------------------------------------------------- monitoring: configurable thresholds, structured alerts
def test_structured_drift_alerts_fire_on_the_shifted_batch():
    base = synth.generate_synthetic_credit_data(4000, seed=11)
    shifted = synth.generate_synthetic_credit_data(4000, seed=11, shifted=True)
    cols = ["EXT_SOURCE_3", "AMT_INCOME_TOTAL", "NAME_CONTRACT_TYPE"]
    ref = mon.build_reference(base[cols])
    table = mon.feature_drift(ref, shifted)
    schema = mon.schema_check(cols, shifted, ref)
    alerts = mon.drift_alert_records(table, None, schema)
    codes = {a.code for a in alerts}
    assert "missingness_jump" in codes
    jump = next(a for a in alerts if a.code == "missingness_jump")
    assert "EXT_SOURCE_3" in jump.columns and jump.value >= jump.threshold
    assert mon.drift_alerts(table, None, schema) == [a.message for a in alerts]  # v1 string interface unchanged
    # a looser missingness threshold silences that alert
    loose = mon.drift_alert_records(table, None, schema, mon.DriftThresholds(missing_change=0.9))
    assert "missingness_jump" not in {a.code for a in loose}
    json.loads(to_json([a.to_dict() for a in alerts]))


def test_psi_status_uses_configured_thresholds():
    assert mon.psi_status(0.15) == "Moderate shift"
    assert mon.psi_status(0.15, moderate=0.2, significant=0.3) == "Stable"
    assert mon.psi_status(None) == "n/a"


# ---------------------------------------------------------------- JSON, provenance, model card
def test_json_is_always_valid():
    text = to_json({"a": float("nan"), "b": [float("inf"), 1.0], "c": np.float32(2.5), "d": np.arange(2)})
    assert json.loads(text) == {"a": None, "b": [None, 1.0], "c": 2.5, "d": [0, 1]}


def test_provenance_vocabulary_is_stable():
    assert [p.value for p in Provenance] == ["observed", "measured", "model", "assumption", "simulated", "synthetic"]


def test_model_card_uses_real_metadata(experiment):
    card = model_card_markdown(experiment.to_artifact().metadata, experiment.comparison_table())
    assert experiment.experiment_id in card and "Out-of-scope" in card
    assert f"{experiment.metrics[experiment.selected]['test']['roc_auc']:.4g}" in card
    assert "SYNTHETIC" in card
