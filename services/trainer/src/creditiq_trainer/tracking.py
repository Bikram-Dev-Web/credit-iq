"""MLflow logging, model registration and alias management.

Every run logs params, per-split metrics for every model, plots, the model card and the data-quality
report. The selected model is logged as an MLflow pyfunc model whose artifacts contain the full
model-version package (see ``creditiq_core.models.package``) and registered under ``creditiq-pd``.

Aliases, not stages (MLflow stages are deprecated): ``staging`` for a new candidate, ``production``
for the serving version and ``previous`` for the version production pointed to before the last promotion,
which makes rollback a single alias move.
"""
from __future__ import annotations

import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from mlflow.pyfunc.model import PythonModel
from mlflow.tracking import MlflowClient

from creditiq_core.models import package as pkg
from creditiq_core.models.training import ExperimentResult
from creditiq_trainer.config import TrainingFileConfig
from creditiq_trainer.package_writer import run_tags, write_package
from creditiq_trainer.pipeline import PreparedData

DEFAULT_TRACKING_URI = "sqlite:///mlruns/mlflow.db"
DEFAULT_EXPERIMENT = "creditiq"
ALIASES = ("staging", "production", "previous")
METRIC_KEYS = ("roc_auc", "gini", "pr_auc", "brier", "log_loss", "ks", "precision", "recall", "specificity", "f1",
               "balanced_accuracy", "flag_rate", "prevalence", "mean_pd", "lift_top10")


class CreditIQPyfunc(PythonModel):
    """Generic MLflow interface: ``predict`` returns calibrated PD and raw score per row.

    The scoring API does not go through pyfunc; it loads the package files directly so that it has the
    schema, explainer inputs and monitoring reference as well.
    """

    def load_context(self, context: Any) -> None:
        path = Path(context.artifacts["package"]) / pkg.ARTIFACT_FILE
        self.artifact = pkg.artifact_from_bytes(path.read_bytes())

    def predict(self, context: Any, model_input: pd.DataFrame, params: dict[str, Any] | None = None) -> pd.DataFrame:
        return pd.DataFrame({"pd": self.artifact.predict_pd(model_input),
                             "raw_score": self.artifact.predict_score(model_input)})


def configure(tracking_uri: str | None = None, experiment: str | None = None) -> str:
    """Point MLflow at the tracking server (or the local SQLite store) and select the experiment."""
    uri = tracking_uri or os.environ.get("MLFLOW_TRACKING_URI") or DEFAULT_TRACKING_URI
    if uri.startswith("sqlite:///"):
        Path(uri.removeprefix("sqlite:///")).parent.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(uri)
    name = experiment or os.environ.get("MLFLOW_EXPERIMENT_NAME") or DEFAULT_EXPERIMENT
    if mlflow.get_experiment_by_name(name) is None:
        location = None
        if uri.startswith("sqlite:///"):  # keep local artifacts beside the database, not in the current directory
            location = (Path(uri.removeprefix("sqlite:///")).parent / "artifacts").resolve().as_uri()
        mlflow.create_experiment(name, artifact_location=location)
    mlflow.set_experiment(name)
    return uri


def _metrics(result: ExperimentResult) -> dict[str, float]:
    out: dict[str, float] = {}
    for key in result.models:
        for split in ("train", "validation", "test"):
            for m in METRIC_KEYS:
                v = result.metrics[key][split].get(m)
                if v is not None and np.isfinite(v):
                    out[f"{key}.{split}.{m}"] = float(v)
        out[f"{key}.threshold"] = float(result.thresholds[key])
    sel = result.selected
    for m in METRIC_KEYS:  # the selected model's test metrics, unprefixed, for the run list
        v = result.metrics[sel]["test"].get(m)
        if v is not None and np.isfinite(v):
            out[f"test_{m}"] = float(v)
    return out


def _params(cfg: TrainingFileConfig, data: PreparedData, result: ExperimentResult) -> dict[str, Any]:
    p: dict[str, Any] = {
        "dataset.path": cfg.dataset.path, "dataset.kind": cfg.dataset.kind, "dataset.rows": len(data.df),
        "target.column": cfg.target.column, "target.default_values": cfg.target.default_values,
        "split.strategy": cfg.split.strategy, "split.validation": cfg.split.validation, "split.test": cfg.split.test,
        "sample_size": cfg.sample_size, "models": ",".join(cfg.models), "gbm_backend": cfg.gbm_backend,
        "class_weight": cfg.class_weight, "calibration": cfg.calibration, "threshold.rule": cfg.threshold.rule,
        "selection_metric": cfg.selection_metric, "tune": cfg.tune, "seed": cfg.seed,
        "leakage_guard": cfg.leakage_guard.enabled, "leakage_guard.auc": cfg.leakage_guard.auc_threshold,
        "n_predictors": len(result.predictors), "selected_model": result.selected,
        "leakage_guard.excluded": ",".join(result.leakage_flags) or "none",
        "sensitive_columns": ",".join(data.sensitive) or "none",
    }
    for key, model in result.models.items():
        for name, value in model.params.items():
            p[f"{key}.{name}"] = value
    return p


@dataclass
class TrainingRun:
    run_id: str
    model_name: str | None
    version: str | None
    alias: str | None
    package_dir: Path


def log_run(cfg: TrainingFileConfig, data: PreparedData, result: ExperimentResult, run_id: str) -> TrainingRun:
    """Log the finished training run to the active MLflow run and register the selected model."""
    tags = run_tags(cfg, data, result)
    mlflow.set_tags(tags)
    mlflow.log_params(_params(cfg, data, result))
    mlflow.log_metrics(_metrics(result))
    with tempfile.TemporaryDirectory() as tmp:
        package_dir = write_package(cfg, data, result, Path(tmp) / "package", {"mlflow_run_id": run_id})
        mlflow.log_artifacts(str(package_dir / "plots"), artifact_path="plots")
        for name in (pkg.MODEL_CARD_FILE, pkg.QUALITY_REPORT_FILE, pkg.EVALUATION_FILE, "config.yaml"):
            mlflow.log_artifact(str(package_dir / name), artifact_path="reports")
        X_sig = result.X_split("validation").head(50)
        # Integers are declared as floats: any input may be missing at scoring time (the pipeline imputes it).
        X_sig = X_sig.astype({c: "float64" for c in X_sig.columns if pd.api.types.is_integer_dtype(X_sig[c])})
        signature = infer_signature(X_sig, pd.DataFrame({"pd": result.pd_split("validation")[:50],
                                                         "raw_score": result.score_split("validation")[:50]}))
        # No input_example: it would copy applicant rows into the model artifact. The signature holds names/types only.
        info = mlflow.pyfunc.log_model(name="model", python_model=CreditIQPyfunc(),
                                       artifacts={"package": str(package_dir)}, signature=signature)
        kept = Path(tempfile.mkdtemp(prefix="creditiq_pkg_"))
        shutil.copytree(package_dir, kept, dirs_exist_ok=True)
    version = alias = None
    model_name = cfg.registry.model_name if cfg.registry.enabled else None
    if model_name:
        mv = mlflow.register_model(info.model_uri, model_name, tags=tags)
        version = str(mv.version)
        client = MlflowClient()
        client.update_model_version(model_name, version, description=f"{cfg.name}: {result.selected_model.label}, "
                                    f"test ROC-AUC {result.metrics[result.selected]['test']['roc_auc']:.4f}"
                                    + (" (SYNTHETIC data)" if data.info.is_synthetic else ""))
        if cfg.registry.alias:
            alias = cfg.registry.alias
            client.set_registered_model_alias(model_name, alias, version)
    return TrainingRun(run_id, model_name, version, alias, kept)


# ---------------------------------------------------------------------------- registry operations

def alias_version(model_name: str, alias: str) -> str | None:
    try:
        return str(MlflowClient().get_model_version_by_alias(model_name, alias).version)
    except mlflow.exceptions.MlflowException:
        return None


def promote(model_name: str, version: str, alias: str = "production") -> dict[str, str | None]:
    """Point ``alias`` at ``version``. Promoting to production remembers the old one as ``previous``."""
    client = MlflowClient()
    client.get_model_version(model_name, version)  # raises if the version does not exist
    old = alias_version(model_name, alias)
    if alias == "production" and old and old != version:
        client.set_registered_model_alias(model_name, "previous", old)
    client.set_registered_model_alias(model_name, alias, version)
    return {"alias": alias, "version": version, "replaced": old}


def rollback(model_name: str) -> dict[str, str | None]:
    """Swap ``production`` and ``previous``."""
    prev = alias_version(model_name, "previous")
    if prev is None:
        raise ValueError("There is no 'previous' version to roll back to.")
    return promote(model_name, prev, "production")


def list_versions(model_name: str) -> list[dict[str, Any]]:
    client = MlflowClient()
    try:
        registered = client.get_registered_model(model_name)
        versions = client.search_model_versions(f"name = '{model_name}'")
    except mlflow.exceptions.MlflowException:
        return []
    by_version: dict[str, list[str]] = {}
    for alias, version in (registered.aliases or {}).items():  # aliases live on the registered model
        by_version.setdefault(str(version), []).append(alias)
    out = []
    for mv in sorted(versions, key=lambda v: int(v.version), reverse=True):
        out.append({"version": str(mv.version), "aliases": sorted(by_version.get(str(mv.version), [])),
                    "run_id": mv.run_id, "created": mv.creation_timestamp, "description": mv.description,
                    "tags": dict(mv.tags or {})})
    return out


def download_package(model_name: str, ref: str, dst: Path) -> Path:
    """Download a registered version's package (``ref`` is a version number or an alias)."""
    uri = f"models:/{model_name}/{ref}" if ref.isdigit() else f"models:/{model_name}@{ref}"
    local = Path(mlflow.artifacts.download_artifacts(artifact_uri=uri, dst_path=str(dst)))
    found = next((p.parent for p in local.rglob(pkg.ARTIFACT_FILE)), None)
    if found is None:
        raise FileNotFoundError(f"{uri} does not contain a CreditIQ package")
    return found
