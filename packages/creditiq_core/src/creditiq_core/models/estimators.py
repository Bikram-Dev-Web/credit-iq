"""The three model families and their default hyperparameters.

* ``logreg`` – Logistic Regression (interpretable linear baseline)
* ``random_forest`` – Random Forest (bagged decision trees)
* ``gbm`` – Gradient boosting with an explicit backend: ``lightgbm``, ``xgboost`` or
  ``sklearn_hgb`` (scikit-learn HistGradientBoosting). ``auto`` picks the first one that is
  installed, as v1 did. The training config should name a backend so that the same config
  always produces the same kind of model; a named backend that is not installed is an error,
  not a silent fallback.
"""
from __future__ import annotations

import importlib.util
from typing import Any, Literal

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

ModelKey = Literal["logreg", "random_forest", "gbm"]
GbmBackend = Literal["lightgbm", "xgboost", "sklearn_hgb"]

MODEL_KEYS: tuple[str, ...] = ("logreg", "random_forest", "gbm")
GBM_BACKENDS: tuple[str, ...] = ("lightgbm", "xgboost", "sklearn_hgb")

BACKEND_LABELS = {
    "lightgbm": "Gradient Boosting (LightGBM)",
    "xgboost": "Gradient Boosting (XGBoost)",
    "sklearn_hgb": "Gradient Boosting (scikit-learn HistGradientBoosting — fallback)",
}


def _has(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


def boosting_backend() -> str:
    """Which gradient-boosting implementation ``auto`` resolves to on this machine."""
    if _has("lightgbm"):
        return "lightgbm"
    if _has("xgboost"):
        return "xgboost"
    return "sklearn_hgb"


def resolve_backend(requested: str = "auto") -> str:
    """Turn a configured backend into a concrete one, failing loudly if it is unavailable."""
    if requested == "auto":
        return boosting_backend()
    if requested not in GBM_BACKENDS:
        raise ValueError(f"Unknown gradient-boosting backend '{requested}'. Choose one of {', '.join(GBM_BACKENDS)} "
                         "or 'auto'.")
    if requested != "sklearn_hgb" and not _has(requested):
        raise ValueError(f"The training config asks for the '{requested}' backend, but the '{requested}' package is "
                         "not installed. Install it or choose another backend explicitly.")
    return requested


def model_label(key: str, backend: str = "auto") -> str:
    if key == "logreg":
        return "Logistic Regression"
    if key == "random_forest":
        return "Random Forest"
    if key == "gbm":
        return BACKEND_LABELS[resolve_backend(backend)]
    raise ValueError(f"Unknown model key '{key}'.")


def default_params(key: str, backend: str = "auto") -> dict[str, Any]:
    if key == "logreg":
        return {"C": 0.3, "max_iter": 2000}
    if key == "random_forest":
        return {"n_estimators": 200, "max_depth": 12, "min_samples_leaf": 20, "max_features": "sqrt"}
    if key == "gbm":
        b = resolve_backend(backend)
        if b == "lightgbm":
            # Deliberately regularised for 10k–60k-row laptop samples; tune on validation for larger data.
            return {"n_estimators": 400, "learning_rate": 0.03, "num_leaves": 15, "min_child_samples": 100,
                    "subsample": 0.8, "subsample_freq": 1, "colsample_bytree": 0.7, "reg_lambda": 5.0}
        if b == "xgboost":
            return {"n_estimators": 400, "learning_rate": 0.03, "max_depth": 4, "min_child_weight": 5, "subsample": 0.8,
                    "colsample_bytree": 0.7, "reg_lambda": 5.0}
        return {"max_iter": 300, "learning_rate": 0.05, "max_leaf_nodes": 15, "min_samples_leaf": 80,
                "l2_regularization": 5.0}
    raise ValueError(f"Unknown model key '{key}'.")


TUNING_GRIDS: dict[str, Any] = {
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


def tuning_grid(key: str, backend: str = "auto") -> list[dict[str, Any]]:
    grid = TUNING_GRIDS[key]
    return list(grid[resolve_backend(backend)] if isinstance(grid, dict) else grid)


def build_estimator(key: str, params: dict[str, Any], class_weight: str, seed: int,
                    y_train: np.ndarray | None = None, backend: str = "auto") -> Any:
    cw = "balanced" if class_weight == "balanced" else None
    if key == "logreg":
        return LogisticRegression(class_weight=cw, random_state=seed, **params)
    if key == "random_forest":
        return RandomForestClassifier(class_weight="balanced_subsample" if cw else None, n_jobs=-1,
                                      random_state=seed, **params)
    if key == "gbm":
        b = resolve_backend(backend)
        if b == "lightgbm":
            from lightgbm import LGBMClassifier

            return LGBMClassifier(class_weight=cw, random_state=seed, n_jobs=-1, verbose=-1, **params)
        if b == "xgboost":
            from xgboost import XGBClassifier

            spw = 1.0
            if cw and y_train is not None and y_train.sum() > 0:
                spw = float((len(y_train) - y_train.sum()) / y_train.sum())
            return XGBClassifier(scale_pos_weight=spw, random_state=seed, n_jobs=-1, tree_method="hist",
                                 eval_metric="logloss", **params)
        return HistGradientBoostingClassifier(class_weight=cw, random_state=seed, early_stopping=False, **params)
    raise ValueError(f"Unknown model key '{key}'.")
