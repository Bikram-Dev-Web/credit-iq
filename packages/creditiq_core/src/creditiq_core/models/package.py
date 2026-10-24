"""The model-version package: what the trainer writes and the API reads.

A package is a directory with these files (names are the contract between services):

=============================  ==============================================================
``model_artifact.joblib``      :class:`ModelArtifact` (pipeline, calibrator, schema, threshold,
                               monitoring reference). No data rows.
``evaluation_snapshot.parquet`` :class:`EvaluationSnapshot` frame (validation + test rows only)
``snapshot_meta.json``         the snapshot's non-frame fields
``evaluation.json``            curves and metrics per model and split (:func:`evaluation_report`)
``metadata.json``              training metadata (dataset, target, split, config, metrics)
``model_card.md``              generated model card
``quality_report.md``          data-quality report of the training file
=============================  ==============================================================

This module is pure: it turns objects into bytes/dicts and back. Services do the file I/O.
"""
from __future__ import annotations

import io
from typing import Any

import joblib
import numpy as np
import pandas as pd

from creditiq_core import evaluation as ev
from creditiq_core.models.artifact import EvaluationSnapshot, ModelArtifact
from creditiq_core.models.training import ExperimentResult

ARTIFACT_FILE = "model_artifact.joblib"
SNAPSHOT_FILE = "evaluation_snapshot.parquet"
SNAPSHOT_META_FILE = "snapshot_meta.json"
EVALUATION_FILE = "evaluation.json"
METADATA_FILE = "metadata.json"
MODEL_CARD_FILE = "model_card.md"
QUALITY_REPORT_FILE = "quality_report.md"

PACKAGE_FILES = (ARTIFACT_FILE, SNAPSHOT_FILE, SNAPSHOT_META_FILE, EVALUATION_FILE, METADATA_FILE, MODEL_CARD_FILE)


def artifact_to_bytes(artifact: ModelArtifact) -> bytes:
    buf = io.BytesIO()
    joblib.dump(artifact, buf, compress=3)
    return buf.getvalue()


def artifact_from_bytes(data: bytes) -> ModelArtifact:
    """Deserialize a model artifact. joblib uses pickle: only load packages from a trusted registry."""
    obj = joblib.load(io.BytesIO(data))
    if not isinstance(obj, ModelArtifact):
        raise ValueError("This file is not a CreditIQ model artifact.")
    return obj


def snapshot_meta(snapshot: EvaluationSnapshot) -> dict[str, Any]:
    return {"model_key": snapshot.model_key, "threshold": snapshot.threshold, "exposure_col": snapshot.exposure_col,
            "date_col": snapshot.date_col, "sensitive_cols": list(snapshot.sensitive_cols),
            "predictors": list(snapshot.predictors)}


def snapshot_from_parts(frame: pd.DataFrame, meta: dict[str, Any]) -> EvaluationSnapshot:
    return EvaluationSnapshot(frame=frame, model_key=meta["model_key"], threshold=float(meta["threshold"]),
                              exposure_col=meta.get("exposure_col"), date_col=meta.get("date_col"),
                              sensitive_cols=list(meta.get("sensitive_cols", [])),
                              predictors=list(meta.get("predictors", [])))


def _records(df: pd.DataFrame) -> list[dict[str, Any]]:
    clean = df.replace([np.inf, -np.inf], np.nan)
    rows = clean.astype(object).where(clean.notna(), None).to_dict(orient="records")
    return [{str(k): v for k, v in row.items()} for row in rows]


def evaluation_report(result: ExperimentResult) -> dict[str, Any]:
    """Everything the Models page needs, computed once at training time.

    The threshold sweep is computed on the VALIDATION split only: thresholds are chosen there, and the test split
    is reported, never tuned on (the v1 threshold-explorer rule).
    """
    models: dict[str, Any] = {}
    for key, model in result.models.items():
        per_split: dict[str, Any] = {}
        for split in ("validation", "test"):
            y, p = result.y_split(split), result.pd_split(split, key)
            cal = ev.calibration_table(y, p)
            per_split[split] = {
                "metrics": result.metrics[key][split],
                "roc": _records(ev.roc_points(y, p)),
                "pr": _records(ev.pr_points(y, p)),
                "calibration": _records(cal),
                "ece": ev.expected_calibration_error(cal),
                "deciles": _records(ev.decile_table(y, p)),
            }
        y_val, p_val = result.y_split("validation"), result.pd_split("validation", key)
        models[key] = {
            "label": model.label, "backend": model.backend, "params": model.params,
            "calibration": model.calibrator.describe(), "is_calibrated": model.is_calibrated,
            "threshold": result.thresholds[key], "fit_seconds": model.fit_seconds,
            "train_metrics": result.metrics[key]["train"],
            "splits": per_split,
            "threshold_sweep_validation": _records(ev.threshold_sweep(y_val, p_val)),
        }
    comparison = result.comparison_table()
    return {"experiment_id": result.experiment_id, "selected": result.selected,
            "threshold_rule": result.config.threshold_rule, "comparison": _records(comparison), "models": models,
            "split_sizes": result.split.sizes(), "warnings": list(result.warnings),
            "leakage_guard_excluded": dict(result.leakage_flags)}
