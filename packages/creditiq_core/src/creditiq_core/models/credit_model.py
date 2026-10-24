"""A fitted scoring pipeline plus its validation-fitted calibrator, and the feature schema."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from creditiq_core.calibration import ProbabilityCalibrator
from creditiq_core.data.validation import KIND_NUMERIC, infer_kind
from creditiq_core.features.engineering import DomainFeatureBuilder, FrameCleaner
from creditiq_core.features.preprocessing import map_transformed_to_original


def build_feature_schema(X_train: pd.DataFrame, max_categories: int = 60) -> dict[str, dict[str, Any]]:
    """Summary of each raw predictor computed on the TRAINING split only.

    Drives the dynamic scoring form, request validation and explanation baselines.
    """
    schema: dict[str, dict[str, Any]] = {}
    for col in X_train.columns:
        s = X_train[col]
        kind = infer_kind(s)
        entry: dict[str, Any] = {"kind": "numeric" if kind == KIND_NUMERIC else "categorical",
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
                          "n_categories": len(counts)})
        schema[str(col)] = entry
    return schema


class CreditModel:
    """A fitted pipeline plus its validation-fitted calibrator."""

    def __init__(self, key: str, label: str, pipeline: Pipeline, calibrator: ProbabilityCalibrator,
                 params: dict[str, Any], input_columns: list[str], fit_seconds: float, backend: str):
        self.key = key
        self.label = label
        self.pipeline = pipeline
        self.calibrator = calibrator
        self.params = params
        self.input_columns = list(input_columns)
        self.fit_seconds = fit_seconds
        self.backend = backend

    # -- inference ---------------------------------------------------------
    def _frame(self, X: pd.DataFrame | pd.Series | dict[str, Any]) -> pd.DataFrame:
        if isinstance(X, pd.Series):
            X = X.to_frame().T
        if isinstance(X, dict):
            X = pd.DataFrame([X])
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Model input must be a pandas DataFrame, Series or dict.")
        if len(X) == 0:
            raise ValueError("No rows to score.")
        return X

    def check_input(self, X: pd.DataFrame | pd.Series | dict[str, Any]) -> list[str]:
        X = self._frame(X)
        msgs = []
        missing = [c for c in self.input_columns if c not in X.columns]
        if missing:
            msgs.append(f"{len(missing)} model input column(s) are absent and will be treated as missing: "
                        + ", ".join(missing[:8]) + (" …" if len(missing) > 8 else ""))
        if len(missing) == len(self.input_columns):
            raise ValueError("None of the model's input columns are present — this data is incompatible with the model.")
        return msgs

    def predict_score(self, X: pd.DataFrame | pd.Series | dict[str, Any]) -> np.ndarray:
        """Raw model score (``predict_proba``). Not necessarily a calibrated probability."""
        X = self._frame(X)
        self.check_input(X)
        return np.asarray(self.pipeline.predict_proba(X)[:, 1], dtype=float)

    def predict_pd(self, X: pd.DataFrame | pd.Series | dict[str, Any]) -> np.ndarray:
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
    def estimator(self) -> Any:
        return self.pipeline.named_steps["model"]

    def transform(self, X: pd.DataFrame | pd.Series | dict[str, Any]) -> np.ndarray:
        X = self._frame(X)
        pre = Pipeline(self.pipeline.steps[:-1])
        return np.asarray(pre.transform(X))

    def transformed_feature_names(self) -> list[str]:
        return [str(n) for n in self.pipeline.named_steps["preprocess"].get_feature_names_out()]

    def feature_origin_map(self) -> dict[str, str]:
        cleaner: FrameCleaner = self.pipeline.named_steps["clean"]
        builder: DomainFeatureBuilder = self.pipeline.named_steps["features"]
        numeric = list(cleaner.numeric_columns_) + list(builder.features_)
        return map_transformed_to_original(self.transformed_feature_names(), numeric, cleaner.categorical_columns_)

    def derived_features(self) -> list[str]:
        return list(self.pipeline.named_steps["features"].features_)
