"""Pipeline-safe input cleaning and domain feature construction.

Both transformers are stateless with respect to the *target* and only learn the
input schema during ``fit`` (on the training split), so they cannot leak test
information. They live inside the saved model pipeline, which means the borrower
simulator and monitoring pages apply exactly the same transformations.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from creditiq_core.data.validation import KIND_NUMERIC, infer_kind

HOME_CREDIT_EMPLOYED_SENTINEL = 365243


class FrameCleaner(BaseEstimator, TransformerMixin):
    """Enforce the training schema at inference time.

    * Adds columns missing from new data as all-missing (recorded in
      ``last_missing_columns_``) and drops unexpected extra columns.
    * Coerces numeric columns with ``pd.to_numeric`` and turns ±inf into NaN.
    * Casts categorical columns to plain Python strings (object dtype) so the
      encoder sees consistent values; missing stays ``np.nan``.
    """

    def fit(self, X: pd.DataFrame, y: Any = None) -> FrameCleaner:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("FrameCleaner expects a pandas DataFrame.")
        self.columns_ = list(X.columns)
        self.numeric_columns_ = [c for c in X.columns if infer_kind(X[c]) == KIND_NUMERIC]
        self.categorical_columns_ = [c for c in X.columns if c not in self.numeric_columns_]
        self.last_missing_columns_: list[str] = []
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.columns_[: np.shape(X)[1]])
        missing = [c for c in self.columns_ if c not in X.columns]
        self.last_missing_columns_ = missing
        out: dict[str, np.ndarray] = {}
        for col in self.columns_:
            if col in missing:
                out[col] = np.full(len(X), np.nan, dtype=float if col in self.numeric_columns_ else object)
                continue
            s = X[col]
            if col in self.numeric_columns_:
                v = pd.to_numeric(s, errors="coerce").to_numpy(dtype=float, na_value=np.nan, copy=True)
                v[~np.isfinite(v)] = np.nan
                out[col] = v
            else:
                arr = s.to_numpy(dtype=object, na_value=np.nan) if hasattr(s, "to_numpy") else np.asarray(s, dtype=object)
                mask = pd.isna(arr)
                arr = np.array([str(v) for v in arr], dtype=object)
                arr[mask] = np.nan
                out[col] = arr
        return pd.DataFrame(out, index=X.index)

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        return np.asarray(self.columns_, dtype=object)


def _safe_div(a: pd.Series, b: pd.Series) -> np.ndarray:
    num = pd.to_numeric(a, errors="coerce").to_numpy(dtype=float, na_value=np.nan)
    den = pd.to_numeric(b, errors="coerce").to_numpy(dtype=float, na_value=np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = num / den
    out[~np.isfinite(out)] = np.nan
    return out


# name -> (required input columns, description)
DOMAIN_FEATURES = {
    "CREDIT_INCOME_RATIO": (("AMT_CREDIT", "AMT_INCOME_TOTAL"), "Loan amount ÷ annual income (leverage)"),
    "ANNUITY_INCOME_RATIO": (("AMT_ANNUITY", "AMT_INCOME_TOTAL"), "Instalment ÷ income (debt-service burden)"),
    "PAYMENT_RATE": (("AMT_ANNUITY", "AMT_CREDIT"), "Instalment ÷ loan amount (proxy for term)"),
    "GOODS_CREDIT_RATIO": (("AMT_GOODS_PRICE", "AMT_CREDIT"), "Goods price ÷ loan amount"),
    "INCOME_PER_PERSON": (("AMT_INCOME_TOTAL", "CNT_FAM_MEMBERS"), "Income ÷ family members"),
    "DAYS_EMPLOYED_ANOM": (("DAYS_EMPLOYED",), "1 if DAYS_EMPLOYED holds the 365243 placeholder"),
    "EMPLOYED_TO_AGE_RATIO": (("DAYS_EMPLOYED", "DAYS_BIRTH"), "Employment length ÷ age"),
    "EXT_SOURCE_MEAN": (("EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"), "Mean of available external scores"),
    "EXT_SOURCE_MISSING_COUNT": (("EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"), "Number of missing external scores"),
}


class DomainFeatureBuilder(BaseEstimator, TransformerMixin):
    """Add row-wise credit ratios when their source columns are available.

    Ratios are deterministic per row (no fitted statistics), so they are safe to
    compute before or after splitting. Only features whose inputs are among the
    selected predictors are built — e.g. if ``DAYS_BIRTH`` is excluded as a
    sensitive attribute, ``EMPLOYED_TO_AGE_RATIO`` is not created.
    """

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled

    def fit(self, X: pd.DataFrame, y: Any = None) -> DomainFeatureBuilder:
        cols = set(X.columns)
        self.features_: list[str] = []
        if self.enabled:
            for name, (needed, _) in DOMAIN_FEATURES.items():
                if name.startswith("EXT_SOURCE"):
                    if any(c in cols for c in needed):
                        self.features_.append(name)
                elif all(c in cols for c in needed):
                    self.features_.append(name)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        if not self.features_:
            return X
        if "DAYS_EMPLOYED" in X.columns:
            emp = pd.to_numeric(X["DAYS_EMPLOYED"], errors="coerce")
            anom = emp == HOME_CREDIT_EMPLOYED_SENTINEL
            if "DAYS_EMPLOYED_ANOM" in self.features_:
                X["DAYS_EMPLOYED_ANOM"] = anom.astype(float)
            X["DAYS_EMPLOYED"] = emp.where(~anom, np.nan)
        for name in self.features_:
            if name == "CREDIT_INCOME_RATIO":
                X[name] = _safe_div(X["AMT_CREDIT"], X["AMT_INCOME_TOTAL"])
            elif name == "ANNUITY_INCOME_RATIO":
                X[name] = _safe_div(X["AMT_ANNUITY"], X["AMT_INCOME_TOTAL"])
            elif name == "PAYMENT_RATE":
                X[name] = _safe_div(X["AMT_ANNUITY"], X["AMT_CREDIT"])
            elif name == "GOODS_CREDIT_RATIO":
                X[name] = _safe_div(X["AMT_GOODS_PRICE"], X["AMT_CREDIT"])
            elif name == "INCOME_PER_PERSON":
                X[name] = _safe_div(X["AMT_INCOME_TOTAL"], X["CNT_FAM_MEMBERS"])
            elif name == "EMPLOYED_TO_AGE_RATIO":
                X[name] = _safe_div(X["DAYS_EMPLOYED"], X["DAYS_BIRTH"])
            elif name in ("EXT_SOURCE_MEAN", "EXT_SOURCE_MISSING_COUNT"):
                ext_cols = [c for c in ("EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3") if c in X.columns]
                ext = X[ext_cols].apply(pd.to_numeric, errors="coerce")
                if name == "EXT_SOURCE_MEAN":
                    X[name] = ext.mean(axis=1, skipna=True).astype(float)
                else:
                    X[name] = ext.isna().sum(axis=1).astype(float)
        return X

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        return np.asarray(list(input_features or []) + self.features_, dtype=object)
