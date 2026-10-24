"""Population stability and data-quality monitoring against the training reference.

The reference stores only binned proportions (no borrower rows), computed from
the TRAINING split for features and from out-of-sample VALIDATION scores for the
PD distribution.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config
from .data_validation import KIND_NUMERIC, infer_kind

MISSING = "__MISSING__"
OTHER = "__OTHER__"


def psi_from_props(expected: np.ndarray, actual: np.ndarray, eps: float = 1e-4) -> float:
    """Population Stability Index: Σ (a − e) · ln(a / e)."""
    e = np.clip(np.asarray(expected, dtype=float), eps, None)
    a = np.clip(np.asarray(actual, dtype=float), eps, None)
    e, a = e / e.sum(), a / a.sum()
    return float(np.sum((a - e) * np.log(a / e)))


def psi_status(value: float | None) -> str:
    if value is None or not np.isfinite(value):
        return "n/a"
    if value >= config.PSI_SIGNIFICANT:
        return "Significant shift"
    if value >= config.PSI_MODERATE:
        return "Moderate shift"
    return "Stable"


def _numeric_edges(values: pd.Series, n_bins: int) -> list[float]:
    v = values.dropna()
    if v.empty:
        return []
    edges = np.unique(np.quantile(v, np.linspace(0, 1, n_bins + 1)))
    return edges[1:-1].tolist()  # interior cut points; outer bins are open-ended


def _numeric_props(values: pd.Series, cuts: list[float]) -> np.ndarray:
    v = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan)
    missing = v.isna().to_numpy()
    bins = np.searchsorted(np.asarray(cuts), v[~missing].to_numpy(), side="right")
    counts = np.bincount(bins, minlength=len(cuts) + 1).astype(float)
    counts = np.append(counts, missing.sum())
    total = counts.sum()
    return counts / total if total else counts


def _categorical_props(values: pd.Series, categories: list[str]) -> np.ndarray:
    s = values.astype(object)
    missing = s.isna()
    keys = s[~missing].astype(str)
    counts = [float((keys == c).sum()) for c in categories]
    counts.append(float((~keys.isin(categories)).sum()))
    counts.append(float(missing.sum()))
    arr = np.asarray(counts)
    return arr / arr.sum() if arr.sum() else arr


def build_score_reference(pd_values: np.ndarray, n_bins: int = 10) -> dict:
    s = pd.Series(np.asarray(pd_values, dtype=float))
    cuts = _numeric_edges(s, n_bins)
    return {"cuts": cuts, "props": _numeric_props(s, cuts).tolist(), "mean": float(s.mean()),
            "quantiles": s.quantile([0.1, 0.25, 0.5, 0.75, 0.9]).round(6).tolist()}


def build_reference(X_train: pd.DataFrame, raw_scores_val=None, pd_val=None, n_bins: int = 10) -> dict:
    features = {}
    for col in X_train.columns:
        s = X_train[col]
        if infer_kind(s) == KIND_NUMERIC:
            v = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)
            cuts = _numeric_edges(v, n_bins)
            features[col] = {"kind": "numeric", "cuts": cuts, "props": _numeric_props(v, cuts).tolist(),
                             "missing_rate": float(v.isna().mean()),
                             "mean": float(v.mean()) if v.notna().any() else None}
        else:
            counts = s.dropna().astype(str).value_counts()
            cats = counts.index[:20].tolist()
            features[col] = {"kind": "categorical", "categories": cats,
                             "props": _categorical_props(s, cats).tolist(),
                             "missing_rate": float(s.isna().mean())}
    ref = {"features": features, "n_rows": int(len(X_train))}
    if pd_val is not None:
        ref["score"] = build_score_reference(pd_val, n_bins)
    return ref


def schema_check(input_columns: list[str], df_new: pd.DataFrame, reference: dict) -> dict:
    missing = [c for c in input_columns if c not in df_new.columns]
    type_changes = []
    for col in input_columns:
        if col in df_new.columns and col in reference.get("features", {}):
            expected = reference["features"][col]["kind"]
            actual = "numeric" if infer_kind(df_new[col]) == KIND_NUMERIC else "categorical"
            if df_new[col].isna().all():
                actual = expected
            if expected != actual:
                type_changes.append({"column": col, "expected": expected, "found": actual})
    return {"missing_columns": missing, "type_changes": type_changes,
            "n_input_columns": len(input_columns), "n_present": len(input_columns) - len(missing)}


def feature_drift(reference: dict, df_new: pd.DataFrame) -> pd.DataFrame:
    """PSI and missingness change per model input column."""
    rows = []
    for col, ref in reference.get("features", {}).items():
        if col not in df_new.columns:
            rows.append({"column": col, "kind": ref["kind"], "psi": None, "status": "Column missing",
                         "ref_missing_rate": ref["missing_rate"], "new_missing_rate": None, "missing_change": None})
            continue
        s = df_new[col]
        if ref["kind"] == "numeric":
            props = _numeric_props(s, ref["cuts"])
            new_missing = float(pd.to_numeric(s, errors="coerce").isna().mean())
        else:
            props = _categorical_props(s, ref["categories"])
            new_missing = float(s.isna().mean())
        value = psi_from_props(np.asarray(ref["props"]), props) if len(props) == len(ref["props"]) else None
        rows.append({"column": col, "kind": ref["kind"], "psi": value, "status": psi_status(value),
                     "ref_missing_rate": ref["missing_rate"], "new_missing_rate": new_missing,
                     "missing_change": new_missing - ref["missing_rate"]})
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values("psi", ascending=False, na_position="first").reset_index(drop=True)
    return out


def score_drift(reference: dict, pd_new: np.ndarray) -> dict | None:
    ref = reference.get("score")
    if not ref:
        return None
    props = _numeric_props(pd.Series(np.asarray(pd_new, dtype=float)), ref["cuts"])
    value = psi_from_props(np.asarray(ref["props"]), props)
    return {"psi": value, "status": psi_status(value), "ref_mean_pd": ref["mean"],
            "new_mean_pd": float(np.mean(pd_new)), "ref_props": ref["props"], "new_props": props.tolist(),
            "cuts": ref["cuts"]}


def drift_alerts(feature_table: pd.DataFrame, score: dict | None, schema: dict) -> list[str]:
    alerts = []
    if schema["missing_columns"]:
        alerts.append(f"{len(schema['missing_columns'])} model input column(s) are missing from the new data.")
    if schema["type_changes"]:
        alerts.append(f"{len(schema['type_changes'])} column(s) changed type versus training.")
    if not feature_table.empty:
        sig = feature_table[feature_table["status"] == "Significant shift"]
        if len(sig):
            alerts.append(f"{len(sig)} feature(s) show a significant distribution shift (PSI ≥ {config.PSI_SIGNIFICANT}).")
        jumps = feature_table[feature_table["missing_change"].fillna(0).abs() >= 0.10]
        if len(jumps):
            alerts.append(f"{len(jumps)} feature(s) changed missingness by ≥ 10 percentage points.")
    if score and score["psi"] >= config.PSI_MODERATE:
        alerts.append(f"The predicted-PD distribution has shifted (PSI = {score['psi']:.3f}).")
    return alerts
