"""Training run: sample → split → fit on train → calibrate and threshold on validation → evaluate on test.

The algorithm is unchanged from v1. Each model is a single scikit-learn ``Pipeline``
(clean → domain features → impute/encode/scale → estimator) fitted on the training split,
plus a probability calibrator fitted on the validation split. The test split is used only
for the final reported metrics.
"""
from __future__ import annotations

import time
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any, ClassVar

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from creditiq_core import defaults
from creditiq_core._frames import iter_rows
from creditiq_core.calibration import ProbabilityCalibrator
from creditiq_core.data.validation import TargetMapping, apply_target_mapping, leakage_scan
from creditiq_core.evaluation import classification_metrics, select_threshold
from creditiq_core.features.engineering import DomainFeatureBuilder, FrameCleaner
from creditiq_core.features.preprocessing import SplitResult, build_preprocessor, make_splits, stratified_sample
from creditiq_core.models.artifact import SNAPSHOT_SPLITS, EvaluationSnapshot, ModelArtifact
from creditiq_core.models.credit_model import CreditModel, build_feature_schema
from creditiq_core.models.estimators import (
    MODEL_KEYS,
    build_estimator,
    default_params,
    model_label,
    resolve_backend,
    tuning_grid,
)
from creditiq_core.monitoring import build_reference, build_score_reference

ProgressFn = Callable[[float, str], None]


def build_pipeline(key: str, params: dict[str, Any], cfg: TrainingConfig, y_train: np.ndarray) -> Pipeline:
    return Pipeline([
        ("clean", FrameCleaner()),
        ("features", DomainFeatureBuilder(enabled=cfg.use_domain_features)),
        ("preprocess", build_preprocessor(scale_numeric=(key == "logreg"), max_categories=cfg.max_categories)),
        ("model", build_estimator(key, params, cfg.class_weight, cfg.seed, y_train, cfg.gbm_backend)),
    ])


# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------

@dataclass
class TrainingConfig:
    model_keys: list[str] = field(default_factory=lambda: list(MODEL_KEYS))
    sample_size: int | None = defaults.DEFAULT_TRAIN_SAMPLE
    split_strategy: str = "stratified"
    val_size: float = 0.2
    test_size: float = 0.2
    seed: int = defaults.DEFAULT_SEED
    class_weight: str = "balanced"
    calibration: str = "sigmoid"
    threshold_rule: str = "max_f1"
    manual_threshold: float = 0.5
    selection_metric: str = "roc_auc"
    tune: bool = False
    use_domain_features: bool = True
    max_categories: int = 30
    params: dict[str, dict[str, Any]] = field(default_factory=dict)
    gbm_backend: str = "auto"  # "lightgbm" | "xgboost" | "sklearn_hgb" | "auto" (first installed, as in v1)
    leakage_guard: bool = True  # drop predictors that alone separate the TRAINING labels almost perfectly
    leakage_auc_threshold: float = 0.90

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


SELECTION_METRICS = {"roc_auc": ("ROC-AUC", True), "pr_auc": ("PR-AUC", True),
                     "brier": ("Brier score", False), "log_loss": ("Log loss", False)}


@dataclass
class ExperimentResult:
    experiment_id: str
    created_at: str
    config: TrainingConfig
    dataset: dict[str, Any]
    target: dict[str, Any]
    predictors: list[str]
    excluded: dict[str, str]
    date_col: str | None
    exposure_col: str | None
    sensitive_cols: list[str]
    frame: pd.DataFrame  # sampled, labelled rows (all original columns)
    y: pd.Series
    split: SplitResult
    models: dict[str, CreditModel]
    scores: dict[str, dict[str, np.ndarray]]  # key -> split -> raw score array
    pds: dict[str, dict[str, np.ndarray]]  # key -> split -> calibrated PD array
    thresholds: dict[str, float]
    metrics: dict[str, dict[str, dict[str, Any]]]  # key -> split -> metric dict
    selected: str
    schema: dict[str, dict[str, Any]]
    reference: dict[str, Any]
    tuning: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    leakage_flags: dict[str, float] = field(default_factory=dict)

    SPLITS: ClassVar[tuple[str, ...]] = ("train", "validation", "test")

    def idx(self, split: str) -> np.ndarray:
        return {"train": self.split.train_idx, "validation": self.split.val_idx, "test": self.split.test_idx}[split]

    def y_split(self, split: str) -> np.ndarray:
        return self.y.to_numpy()[self.idx(split)]

    def frame_split(self, split: str) -> pd.DataFrame:
        return self.frame.iloc[self.idx(split)]

    def X_split(self, split: str) -> pd.DataFrame:
        return self.frame_split(split)[self.predictors]

    def pd_split(self, split: str, key: str | None = None) -> np.ndarray:
        return self.pds[key or self.selected][split]

    def score_split(self, split: str, key: str | None = None) -> np.ndarray:
        return self.scores[key or self.selected][split]

    @property
    def selected_model(self) -> CreditModel:
        return self.models[self.selected]

    def comparison_table(self) -> pd.DataFrame:
        rows = []
        for key, model in self.models.items():
            mv, mt, mtr = self.metrics[key]["validation"], self.metrics[key]["test"], self.metrics[key]["train"]
            rows.append({
                "model": model.label,
                "key": key,
                "selected": key == self.selected,
                "val_roc_auc": mv["roc_auc"], "val_pr_auc": mv["pr_auc"], "val_brier": mv["brier"],
                "test_roc_auc": mt["roc_auc"], "test_gini": mt["gini"], "test_pr_auc": mt["pr_auc"],
                "test_brier": mt["brier"], "test_log_loss": mt["log_loss"], "test_ks": mt["ks"],
                "threshold": self.thresholds[key], "test_recall": mt["recall"],
                "test_precision": mt["precision"], "test_specificity": mt["specificity"],
                "test_f1": mt["f1"], "test_balanced_accuracy": mt["balanced_accuracy"],
                "train_roc_auc": mtr["roc_auc"],
                "overfit_gap_auc": None if mtr["roc_auc"] is None or mt["roc_auc"] is None else mtr["roc_auc"] - mt["roc_auc"],
                "calibration": model.calibrator.fitted_method_, "fit_seconds": model.fit_seconds,
            })
        return pd.DataFrame(rows)

    def metadata(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "created_at": self.created_at,
            "dataset": self.dataset,
            "target": self.target,
            "predictors": self.predictors,
            "excluded_columns": self.excluded,
            "date_column": self.date_col,
            "exposure_column": self.exposure_col,
            "sensitive_columns_for_audit": self.sensitive_cols,
            "split": self.split.to_dict(),
            "config": self.config.to_dict(),
            "models": {k: {"label": m.label, "backend": m.backend, "params": m.params,
                           "calibration": m.calibrator.describe(), "threshold": self.thresholds[k],
                           "fit_seconds": m.fit_seconds,
                           "derived_features": m.derived_features()} for k, m in self.models.items()},
            "selected_model": self.selected,
            "metrics": self.metrics,
            "tuning": self.tuning,
            "leakage_guard_excluded": self.leakage_flags,
            "calibrated": {k: m.is_calibrated for k, m in self.models.items()},
            "warnings": self.warnings,
        }

    def test_predictions(self) -> pd.DataFrame:
        """Held-out predictions with the observed label (for export)."""
        f = self.frame_split("test")
        id_col = next((c for c in self.excluded if self.excluded[c] == "identifier"), None)
        out = pd.DataFrame(index=range(len(f)))
        if id_col and id_col in f.columns:
            out[id_col] = f[id_col].to_numpy()
        out["observed_default (historical label)"] = self.y_split("test")
        for key, model in self.models.items():
            out[f"{key}_raw_score"] = self.scores[key]["test"]
            out[f"{key}_pd"] = self.pds[key]["test"]
            out[f"{key}_flagged"] = (self.pds[key]["test"] >= self.thresholds[key]).astype(int)
        return out

    def to_artifact(self, key: str | None = None) -> ModelArtifact:
        """The deployable model version: pipeline, calibrator, schema, threshold, reference. No data rows."""
        key = key or self.selected
        model = self.models[key]
        meta = self.metadata()
        meta["artifact_model"] = key
        try:
            transformed_mean = np.asarray(model.transform(self.X_split("train")).mean(axis=0)).ravel()
        except (ValueError, TypeError, MemoryError):
            transformed_mean = None
        reference = dict(self.reference)
        reference["score"] = build_score_reference(self.pds[key]["validation"])
        return ModelArtifact(
            model=model, input_schema=self.schema, threshold=self.thresholds[key],
            threshold_rule=self.config.threshold_rule, reference=reference, metadata=meta,
            transformed_mean=transformed_mean,
        )

    def evaluation_snapshot(self, key: str | None = None) -> EvaluationSnapshot:
        """Scored validation and test rows for analytics. Training rows and identifiers are never included."""
        key = key or self.selected
        wanted = [*self.predictors, *([self.exposure_col] if self.exposure_col else []),
                  *([self.date_col] if self.date_col else []), *self.sensitive_cols]
        keep = [c for c in dict.fromkeys(wanted) if c in self.frame.columns and self.excluded.get(c) != "identifier"]
        parts = []
        for split in SNAPSHOT_SPLITS:
            part = self.frame_split(split)[keep].reset_index(drop=True).copy()
            part.insert(0, "raw_score", self.scores[key][split])
            part.insert(0, "pd", self.pds[key][split])
            part.insert(0, "y", self.y_split(split))
            part.insert(0, "split", split)
            parts.append(part)
        return EvaluationSnapshot(
            frame=pd.concat(parts, ignore_index=True), model_key=key, threshold=self.thresholds[key],
            exposure_col=self.exposure_col, date_col=self.date_col,
            sensitive_cols=[c for c in self.sensitive_cols if c in keep], predictors=list(self.predictors),
        )


def prepare_modeling_frame(df: pd.DataFrame, mapping: TargetMapping, max_rows: int | None,
                           seed: int) -> tuple[pd.DataFrame, pd.Series]:
    y = apply_target_mapping(df[mapping.column], mapping.positive_values, mapping.negative_values)
    keep = y.notna()
    frame, y = df.loc[keep], y.loc[keep].astype(int)
    frame, y = stratified_sample(frame, y, max_rows, seed)
    return frame.reset_index(drop=True), y.reset_index(drop=True)


def _fit_one(key: str, params: dict[str, Any], cfg: TrainingConfig, X_tr: pd.DataFrame,
             y_tr: np.ndarray) -> tuple[Pipeline, float]:
    pipe = build_pipeline(key, params, cfg, y_tr)
    t0 = time.perf_counter()
    pipe.fit(X_tr, y_tr)
    return pipe, time.perf_counter() - t0


def run_experiment(
    df: pd.DataFrame,
    dataset_info: dict[str, Any],
    mapping: TargetMapping,
    predictors: list[str],
    cfg: TrainingConfig,
    *,
    date_col: str | None = None,
    exposure_col: str | None = None,
    sensitive_cols: list[str] | None = None,
    excluded: dict[str, str] | None = None,
    progress: ProgressFn | None = None,
    experiment_id: str | None = None,
) -> ExperimentResult:
    """Sample → split → fit (train) → calibrate & pick threshold (validation) → evaluate (test).

    ``experiment_id`` lets the caller supply its own run id (e.g. the MLflow run id).
    """

    def report(frac: float, msg: str) -> None:
        if progress:
            progress(min(max(frac, 0.0), 1.0), msg)

    if not predictors:
        raise ValueError("No predictor columns selected. Choose at least one feature in the Data Center.")
    if mapping.column in predictors:
        raise ValueError("The target column cannot be used as a predictor.")
    if not cfg.model_keys:
        raise ValueError("Select at least one model to train.")
    missing = [c for c in predictors if c not in df.columns]
    if missing:
        raise ValueError(f"Predictor columns not found in the dataset: {missing[:5]}")
    if cfg.split_strategy == "chronological" and not date_col:
        raise ValueError("A chronological split needs a date column (set it in the Data Center).")
    backend = resolve_backend(cfg.gbm_backend) if "gbm" in cfg.model_keys else "sklearn"

    warnings: list[str] = []
    report(0.02, "Preparing labelled sample")
    frame, y = prepare_modeling_frame(df, mapping, cfg.sample_size, cfg.seed)
    if len(frame) < len(df.dropna(subset=[mapping.column])):
        warnings.append(f"Trained on a stratified sample of {len(frame):,} labelled rows (configurable).")
    if y.sum() < 20:
        warnings.append(f"Only {int(y.sum())} default events in the sample — metrics will be very noisy.")

    split = make_splits(y, cfg.split_strategy, cfg.val_size, cfg.test_size, cfg.seed,
                        dates=frame[date_col] if date_col else None)

    # Leakage guard: screened on TRAINING rows only, so the test split never influences feature selection.
    excluded = dict(excluded or {})
    leakage_flags: dict[str, float] = {}
    if cfg.leakage_guard:
        report(0.05, "Screening predictors for leakage on the training split")
        train_frame = frame.iloc[split.train_idx]
        scan = leakage_scan(train_frame, pd.Series(y.to_numpy()[split.train_idx], index=train_frame.index), predictors,
                            sample_size=20_000, flag_threshold=cfg.leakage_auc_threshold, seed=cfg.seed)
        flagged = scan[scan["assessment"].str.startswith("SUSPICIOUS")]
        leakage_flags = {str(r.column): float(r.single_feature_auc) for r in iter_rows(flagged)}
        if leakage_flags:
            predictors = [c for c in predictors if c not in leakage_flags]
            for c, auc in leakage_flags.items():
                excluded[c] = "leakage (training-split screen)"
            warnings.append("Excluded by the leakage guard (single-feature AUC on training rows ≥ "
                            f"{cfg.leakage_auc_threshold:.2f}): " + ", ".join(f"{c} ({a:.3f})" for c, a in leakage_flags.items()))
        if not predictors:
            raise ValueError("Every predictor was excluded by the leakage guard. Review the feature selection.")
    # Predictors with no observed value in the training rows carry no information and break some imputers.
    empty = [c for c in predictors if frame[c].iloc[split.train_idx].isna().all()]
    if empty:
        predictors = [c for c in predictors if c not in empty]
        for c in empty:
            excluded[c] = "empty in training split"
        warnings.append("Dropped predictors with no observed values in the training rows: " + ", ".join(empty))
        if not predictors:
            raise ValueError("No predictor has any observed value in the training rows.")
    X = frame[predictors]
    y_arr = y.to_numpy()
    idx = {"train": split.train_idx, "validation": split.val_idx, "test": split.test_idx}
    X_s = {k: X.iloc[v] for k, v in idx.items()}
    y_s = {k: y_arr[v] for k, v in idx.items()}

    models, scores, pds, thresholds, metrics, tuning = {}, {}, {}, {}, {}, {}
    n_models = len(cfg.model_keys)
    for i, key in enumerate(cfg.model_keys):
        label = model_label(key, cfg.gbm_backend)
        base = 0.08 + 0.85 * i / n_models
        step = 0.85 / n_models
        params = {**default_params(key, cfg.gbm_backend), **cfg.params.get(key, {})}
        if cfg.tune:
            grid = tuning_grid(key, cfg.gbm_backend)
            trials = []
            for j, override in enumerate(grid):
                report(base + step * 0.6 * j / len(grid), f"Tuning {label}: candidate {j + 1}/{len(grid)}")
                cand = {**params, **override}
                pipe, _ = _fit_one(key, cand, cfg, X_s["train"], y_s["train"])
                m = classification_metrics(y_s["validation"], pipe.predict_proba(X_s["validation"])[:, 1])
                trials.append({"params": override, "val_roc_auc": m["roc_auc"]})
            best = max(trials, key=lambda t: t["val_roc_auc"] or 0)
            params = {**params, **best["params"]}
            tuning[key] = {"trials": trials, "chosen": best["params"], "criterion": "validation ROC-AUC"}
        report(base + step * 0.65, f"Training {label} on {len(y_s['train']):,} rows")
        pipe, secs = _fit_one(key, params, cfg, X_s["train"], y_s["train"])
        report(base + step * 0.9, f"Calibrating and evaluating {label}")
        raw = {k: pipe.predict_proba(X_s[k])[:, 1] for k in idx}
        cal = ProbabilityCalibrator(cfg.calibration).fit(raw["validation"], y_s["validation"])
        if cal.warning_:
            warnings.append(f"{label}: {cal.warning_}")
        if cal.fitted_method_ == "none":
            warnings.append(f"{label}: calibration is off, so its outputs are model scores, not probabilities of default "
                            f"(mean validation score {raw['validation'].mean():.1%} vs observed default rate "
                            f"{y_s['validation'].mean():.1%}). Expected-loss figures based on them are distorted.")
        pd_ = {k: cal.transform(v) for k, v in raw.items()}
        thr = select_threshold(y_s["validation"], pd_["validation"], cfg.threshold_rule, cfg.manual_threshold)
        models[key] = CreditModel(key, label, pipe, cal, params, predictors, secs, backend if key == "gbm" else "sklearn")
        scores[key], pds[key], thresholds[key] = raw, pd_, thr
        metrics[key] = {k: classification_metrics(y_s[k], pd_[k], thr) for k in idx}

    metric_name, higher_better = SELECTION_METRICS[cfg.selection_metric]

    def sel_value(k: str) -> float:
        v = metrics[k]["validation"][cfg.selection_metric]
        if v is None:
            return -np.inf
        return v if higher_better else -v

    selected = max(models, key=sel_value)
    report(0.96, "Building feature schema and monitoring reference")
    schema = build_feature_schema(X_s["train"])
    reference = build_reference(X_s["train"], scores[selected]["validation"], pds[selected]["validation"])

    report(1.0, "Done")
    return ExperimentResult(
        experiment_id=experiment_id or uuid.uuid4().hex[:10],
        created_at=datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC"),
        config=cfg, dataset=dataset_info, target=mapping.to_dict(), predictors=list(predictors),
        excluded=excluded, date_col=date_col, exposure_col=exposure_col,
        sensitive_cols=list(sensitive_cols or []), frame=frame, y=y, split=split,
        models=models, scores=scores, pds=pds, thresholds=thresholds, metrics=metrics,
        selected=selected, schema=schema, reference=reference, tuning=tuning, warnings=warnings,
        leakage_flags=leakage_flags,
    )
