"""Model training engine (independent of the UI).

Three genuinely different algorithms are offered:

* ``logreg``  – Logistic Regression (interpretable linear baseline)
* ``random_forest`` – Random Forest (bagged decision trees)
* ``gbm`` – Gradient boosting: LightGBM if installed, else XGBoost if installed,
  else scikit-learn's HistGradientBoostingClassifier. The label always states
  which backend actually ran.

Each model is a single scikit-learn ``Pipeline`` (clean → domain features →
impute/encode/scale → estimator) fitted on the training split, plus a
probability calibrator fitted on the validation split.
"""
from __future__ import annotations

import importlib.util
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from . import config
from .calibration import ProbabilityCalibrator
from .data_validation import KIND_NUMERIC, TargetMapping, apply_target_mapping, infer_kind, leakage_scan
from .evaluation import classification_metrics, select_threshold
from .feature_engineering import DomainFeatureBuilder, FrameCleaner
from .preprocessing import (
    SplitResult,
    build_preprocessor,
    make_splits,
    map_transformed_to_original,
    stratified_sample,
)

ProgressFn = Callable[[float, str], None]

MODEL_KEYS = ("logreg", "random_forest", "gbm")


def _has(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


def boosting_backend() -> str:
    """Which gradient-boosting implementation is available on this machine."""
    if _has("lightgbm"):
        return "lightgbm"
    if _has("xgboost"):
        return "xgboost"
    return "sklearn_hgb"


BACKEND_LABELS = {
    "lightgbm": "Gradient Boosting (LightGBM)",
    "xgboost": "Gradient Boosting (XGBoost)",
    "sklearn_hgb": "Gradient Boosting (scikit-learn HistGradientBoosting — fallback)",
}


def model_label(key: str) -> str:
    if key == "logreg":
        return "Logistic Regression"
    if key == "random_forest":
        return "Random Forest"
    if key == "gbm":
        return BACKEND_LABELS[boosting_backend()]
    raise ValueError(f"Unknown model key '{key}'.")


def default_params(key: str) -> dict:
    if key == "logreg":
        return {"C": 0.3, "max_iter": 2000}
    if key == "random_forest":
        return {"n_estimators": 200, "max_depth": 12, "min_samples_leaf": 20, "max_features": "sqrt"}
    if key == "gbm":
        backend = boosting_backend()
        if backend == "lightgbm":
            # Deliberately regularised for 10k–60k-row laptop samples; tune on validation for larger data.
            return {"n_estimators": 400, "learning_rate": 0.03, "num_leaves": 15, "min_child_samples": 100,
                    "subsample": 0.8, "subsample_freq": 1, "colsample_bytree": 0.7, "reg_lambda": 5.0}
        if backend == "xgboost":
            return {"n_estimators": 400, "learning_rate": 0.03, "max_depth": 4, "min_child_weight": 5, "subsample": 0.8,
                    "colsample_bytree": 0.7, "reg_lambda": 5.0}
        return {"max_iter": 300, "learning_rate": 0.05, "max_leaf_nodes": 15, "min_samples_leaf": 80,
                "l2_regularization": 5.0}
    raise ValueError(f"Unknown model key '{key}'.")


TUNING_GRIDS = {
    "logreg": [{"C": 0.05}, {"C": 0.3}, {"C": 1.0}],
    "random_forest": [{"max_depth": 8, "min_samples_leaf": 50}, {"max_depth": 12, "min_samples_leaf": 20},
                      {"max_depth": None, "min_samples_leaf": 10}],
    "gbm": {
        "lightgbm": [{"num_leaves": 7, "learning_rate": 0.05, "n_estimators": 400},
                     {"num_leaves": 15, "learning_rate": 0.03, "n_estimators": 400},
                     {"num_leaves": 31, "learning_rate": 0.03, "n_estimators": 600}],
        "xgboost": [{"max_depth": 3, "n_estimators": 400}, {"max_depth": 5, "n_estimators": 400},
                    {"max_depth": 7, "n_estimators": 300}],
        "sklearn_hgb": [{"max_leaf_nodes": 15, "max_iter": 300}, {"max_leaf_nodes": 31, "max_iter": 300},
                        {"max_leaf_nodes": 63, "max_iter": 200}],
    },
}


def build_estimator(key: str, params: dict, class_weight: str, seed: int, y_train: np.ndarray | None = None):
    cw = "balanced" if class_weight == "balanced" else None
    if key == "logreg":
        return LogisticRegression(class_weight=cw, random_state=seed, **params)
    if key == "random_forest":
        return RandomForestClassifier(class_weight="balanced_subsample" if cw else None, n_jobs=-1,
                                      random_state=seed, **params)
    if key == "gbm":
        backend = boosting_backend()
        if backend == "lightgbm":
            from lightgbm import LGBMClassifier
            return LGBMClassifier(class_weight=cw, random_state=seed, n_jobs=-1, verbose=-1, **params)
        if backend == "xgboost":
            from xgboost import XGBClassifier
            spw = 1.0
            if cw and y_train is not None and y_train.sum() > 0:
                spw = float((len(y_train) - y_train.sum()) / y_train.sum())
            return XGBClassifier(scale_pos_weight=spw, random_state=seed, n_jobs=-1, tree_method="hist",
                                 eval_metric="logloss", **params)
        return HistGradientBoostingClassifier(class_weight=cw, random_state=seed, early_stopping=False, **params)
    raise ValueError(f"Unknown model key '{key}'.")


# ---------------------------------------------------------------------------
# Feature schema (drives the dynamic simulator form and input validation)
# ---------------------------------------------------------------------------

def build_feature_schema(X_train: pd.DataFrame, max_categories: int = 60) -> dict:
    """Summary of each raw predictor computed on the TRAINING split only."""
    schema: dict[str, dict] = {}
    for col in X_train.columns:
        s = X_train[col]
        kind = infer_kind(s)
        entry: dict = {"kind": "numeric" if kind == KIND_NUMERIC else "categorical",
                       "missing_rate": float(s.isna().mean())}
        if kind == KIND_NUMERIC:
            v = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
            if len(v):
                q = v.quantile([0.01, 0.05, 0.5, 0.95, 0.99])
                entry.update({"min": float(v.min()), "max": float(v.max()), "p01": float(q.loc[0.01]),
                              "p05": float(q.loc[0.05]), "median": float(q.loc[0.5]),
                              "p95": float(q.loc[0.95]), "p99": float(q.loc[0.99]),
                              "is_integer": bool(np.allclose(v, np.round(v)))})
            else:
                entry.update({"min": 0.0, "max": 0.0, "p01": 0.0, "p05": 0.0, "median": 0.0, "p95": 0.0,
                              "p99": 0.0, "is_integer": False})
        else:
            counts = s.dropna().astype(str).value_counts()
            entry.update({"categories": counts.index[:max_categories].tolist(),
                          "mode": counts.index[0] if len(counts) else "",
                          "n_categories": int(len(counts))})
        schema[col] = entry
    return schema


# ---------------------------------------------------------------------------
# Trained model wrapper
# ---------------------------------------------------------------------------

class CreditModel:
    """A fitted pipeline plus its validation-fitted calibrator."""

    def __init__(self, key: str, label: str, pipeline: Pipeline, calibrator: ProbabilityCalibrator,
                 params: dict, input_columns: list[str], fit_seconds: float, backend: str):
        self.key = key
        self.label = label
        self.pipeline = pipeline
        self.calibrator = calibrator
        self.params = params
        self.input_columns = list(input_columns)
        self.fit_seconds = fit_seconds
        self.backend = backend

    # -- inference ---------------------------------------------------------
    def _frame(self, X) -> pd.DataFrame:
        if isinstance(X, pd.Series):
            X = X.to_frame().T
        if isinstance(X, dict):
            X = pd.DataFrame([X])
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Model input must be a pandas DataFrame, Series or dict.")
        if len(X) == 0:
            raise ValueError("No rows to score.")
        return X

    def check_input(self, X) -> list[str]:
        X = self._frame(X)
        msgs = []
        missing = [c for c in self.input_columns if c not in X.columns]
        if missing:
            msgs.append(f"{len(missing)} model input column(s) are absent and will be treated as missing: "
                        + ", ".join(missing[:8]) + (" …" if len(missing) > 8 else ""))
        if len(missing) == len(self.input_columns):
            raise ValueError("None of the model's input columns are present — this data is incompatible with the model.")
        return msgs

    def predict_score(self, X) -> np.ndarray:
        """Raw model score (``predict_proba``). Not necessarily a calibrated probability."""
        X = self._frame(X)
        self.check_input(X)
        return self.pipeline.predict_proba(X)[:, 1]

    def predict_pd(self, X) -> np.ndarray:
        """Calibrated probability of default (validation-fitted calibration).

        If calibration was switched off (or impossible), this returns the raw
        score clipped to [0, 1]; check :attr:`is_calibrated` before treating it as a PD.
        """
        return self.calibrator.transform(self.predict_score(X))

    @property
    def is_calibrated(self) -> bool:
        return getattr(self.calibrator, "fitted_method_", "none") != "none"

    # -- introspection -----------------------------------------------------
    @property
    def estimator(self):
        return self.pipeline.named_steps["model"]

    def transform(self, X) -> np.ndarray:
        X = self._frame(X)
        pre = Pipeline(self.pipeline.steps[:-1])
        return pre.transform(X)

    def transformed_feature_names(self) -> list[str]:
        return [str(n) for n in self.pipeline.named_steps["preprocess"].get_feature_names_out()]

    def feature_origin_map(self) -> dict[str, str]:
        cleaner: FrameCleaner = self.pipeline.named_steps["clean"]
        builder: DomainFeatureBuilder = self.pipeline.named_steps["features"]
        numeric = list(cleaner.numeric_columns_) + list(builder.features_)
        return map_transformed_to_original(self.transformed_feature_names(), numeric, cleaner.categorical_columns_)

    def derived_features(self) -> list[str]:
        return list(self.pipeline.named_steps["features"].features_)


def build_pipeline(key: str, params: dict, cfg: "TrainingConfig", y_train: np.ndarray) -> Pipeline:
    return Pipeline([
        ("clean", FrameCleaner()),
        ("features", DomainFeatureBuilder(enabled=cfg.use_domain_features)),
        ("preprocess", build_preprocessor(scale_numeric=(key == "logreg"), max_categories=cfg.max_categories)),
        ("model", build_estimator(key, params, cfg.class_weight, cfg.seed, y_train)),
    ])


# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------

@dataclass
class TrainingConfig:
    model_keys: list[str] = field(default_factory=lambda: list(MODEL_KEYS))
    sample_size: int | None = config.DEFAULT_TRAIN_SAMPLE
    split_strategy: str = "stratified"
    val_size: float = 0.2
    test_size: float = 0.2
    seed: int = config.DEFAULT_SEED
    class_weight: str = "balanced"
    calibration: str = "sigmoid"
    threshold_rule: str = "max_f1"
    manual_threshold: float = 0.5
    selection_metric: str = "roc_auc"
    tune: bool = False
    use_domain_features: bool = True
    max_categories: int = 30
    params: dict = field(default_factory=dict)
    leakage_guard: bool = True  # drop predictors that alone separate the TRAINING labels almost perfectly
    leakage_auc_threshold: float = 0.90

    def to_dict(self) -> dict:
        return asdict(self)


SELECTION_METRICS = {"roc_auc": ("ROC-AUC", True), "pr_auc": ("PR-AUC", True),
                     "brier": ("Brier score", False), "log_loss": ("Log loss", False)}


@dataclass
class ModelBundle:
    """Everything the simulator / monitoring pages need, without training rows."""

    model: CreditModel
    schema: dict
    threshold: float
    metadata: dict
    eval_val: dict  # {"y": array, "pd": array}
    eval_test: dict
    reference: dict
    transformed_mean: np.ndarray | None = None
    format_version: int = 1


@dataclass
class ExperimentResult:
    experiment_id: str
    created_at: str
    config: TrainingConfig
    dataset: dict
    target: dict
    predictors: list[str]
    excluded: dict
    date_col: str | None
    exposure_col: str | None
    sensitive_cols: list[str]
    frame: pd.DataFrame  # sampled, labelled rows (all original columns)
    y: pd.Series
    split: SplitResult
    models: dict
    scores: dict  # key -> split -> raw score array
    pds: dict  # key -> split -> calibrated PD array
    thresholds: dict
    metrics: dict  # key -> split -> metric dict
    selected: str
    schema: dict
    reference: dict
    tuning: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    leakage_flags: dict = field(default_factory=dict)

    SPLITS = ("train", "validation", "test")

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

    def metadata(self) -> dict:
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

    def bundle(self, key: str | None = None) -> ModelBundle:
        key = key or self.selected
        model = self.models[key]
        meta = self.metadata()
        meta["bundle_model"] = key
        try:
            transformed_mean = np.asarray(model.transform(self.X_split("train")).mean(axis=0)).ravel()
        except (ValueError, TypeError, MemoryError):
            transformed_mean = None
        from .monitoring import build_score_reference

        reference = dict(self.reference)
        reference["score"] = build_score_reference(self.pds[key]["validation"])
        return ModelBundle(
            model=model, schema=self.schema, threshold=self.thresholds[key], metadata=meta,
            eval_val={"y": self.y_split("validation"), "pd": self.pds[key]["validation"]},
            eval_test={"y": self.y_split("test"), "pd": self.pds[key]["test"]},
            reference=reference, transformed_mean=transformed_mean,
        )


def prepare_modeling_frame(df: pd.DataFrame, mapping: TargetMapping, max_rows: int | None, seed: int) -> tuple[pd.DataFrame, pd.Series]:
    y = apply_target_mapping(df[mapping.column], mapping.positive_values, mapping.negative_values)
    keep = y.notna()
    frame, y = df.loc[keep], y.loc[keep].astype(int)
    frame, y = stratified_sample(frame, y, max_rows, seed)
    return frame.reset_index(drop=True), y.reset_index(drop=True)


def _fit_one(key: str, params: dict, cfg: TrainingConfig, X_tr: pd.DataFrame, y_tr: np.ndarray) -> tuple[Pipeline, float]:
    pipe = build_pipeline(key, params, cfg, y_tr)
    t0 = time.perf_counter()
    pipe.fit(X_tr, y_tr)
    return pipe, time.perf_counter() - t0


def run_experiment(
    df: pd.DataFrame,
    dataset_info: dict,
    mapping: TargetMapping,
    predictors: list[str],
    cfg: TrainingConfig,
    *,
    date_col: str | None = None,
    exposure_col: str | None = None,
    sensitive_cols: list[str] | None = None,
    excluded: dict | None = None,
    progress: ProgressFn | None = None,
) -> ExperimentResult:
    """Sample → split → fit (train) → calibrate & pick threshold (validation) → evaluate (test)."""
    from .monitoring import build_reference  # local import avoids a cycle

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
        leakage_flags = {r.column: float(r.single_feature_auc) for r in flagged.itertuples()}
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
        label = model_label(key)
        base = 0.08 + 0.85 * i / n_models
        step = 0.85 / n_models
        params = {**default_params(key), **cfg.params.get(key, {})}
        if cfg.tune:
            grid = TUNING_GRIDS[key]
            if isinstance(grid, dict):
                grid = grid[boosting_backend()]
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
        models[key] = CreditModel(key, label, pipe, cal, params, predictors, secs, boosting_backend() if key == "gbm" else "sklearn")
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
        experiment_id=uuid.uuid4().hex[:10],
        created_at=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        config=cfg, dataset=dataset_info, target=mapping.to_dict(), predictors=list(predictors),
        excluded=excluded, date_col=date_col, exposure_col=exposure_col,
        sensitive_cols=list(sensitive_cols or []), frame=frame, y=y, split=split,
        models=models, scores=scores, pds=pds, thresholds=thresholds, metrics=metrics,
        selected=selected, schema=schema, reference=reference, tuning=tuning, warnings=warnings,
        leakage_flags=leakage_flags,
    )
