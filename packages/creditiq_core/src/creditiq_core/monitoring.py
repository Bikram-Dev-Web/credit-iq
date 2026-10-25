"""Population stability and data-quality monitoring against the training reference.

The reference stores only binned proportions (no borrower rows), computed from
the TRAINING split for features and from out-of-sample VALIDATION scores for the
PD distribution.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from creditiq_core import defaults
from creditiq_core.data.validation import KIND_NUMERIC, infer_kind

MISSING = "__MISSING__"
OTHER = "__OTHER__"


def psi_from_props(expected: ArrayLike, actual: ArrayLike, eps: float = 1e-4) -> float:
    """Population Stability Index: Σ (a − e) · ln(a / e)."""
    e = np.clip(np.asarray(expected, dtype=float), eps, None)
    a = np.clip(np.asarray(actual, dtype=float), eps, None)
    e, a = e / e.sum(), a / a.sum()
    return float(np.sum((a - e) * np.log(a / e)))


def psi_status(value: float | None, moderate: float = defaults.PSI_MODERATE,
               significant: float = defaults.PSI_SIGNIFICANT) -> str:
    if value is None or not np.isfinite(value):
        return "n/a"
    if value >= significant:
        return "Significant shift"
    if value >= moderate:
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
    # Compare at float32 precision: training data may have been down-cast to float32 while a monitoring batch is
    # float64, and values that sit exactly on a cut point (common for discrete measures) would otherwise flip bins.
    bins = np.searchsorted(np.asarray(cuts, dtype=np.float32), v[~missing].to_numpy(dtype=np.float32), side="right")
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


def build_score_reference(pd_values: ArrayLike, n_bins: int = 10) -> dict[str, Any]:
    s = pd.Series(np.asarray(pd_values, dtype=float))
    cuts = _numeric_edges(s, n_bins)
    return {"cuts": cuts, "props": _numeric_props(s, cuts).tolist(), "mean": float(s.mean()),
            "quantiles": s.quantile([0.1, 0.25, 0.5, 0.75, 0.9]).round(6).tolist()}


def build_reference(X_train: pd.DataFrame, raw_scores_val: ArrayLike | None = None, pd_val: ArrayLike | None = None,
                    n_bins: int = 10) -> dict[str, Any]:
    features: dict[str, dict[str, Any]] = {}
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
    ref: dict[str, Any] = {"features": features, "n_rows": int(len(X_train))}
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


def feature_drift(reference: dict, df_new: pd.DataFrame, thresholds: DriftThresholds | None = None) -> pd.DataFrame:
    """PSI and missingness change per model input column."""
    t = thresholds or DriftThresholds()
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
        rows.append({"column": col, "kind": ref["kind"], "psi": value, "status": psi_status(value, t.psi_moderate, t.psi_significant),
                     "ref_missing_rate": ref["missing_rate"], "new_missing_rate": new_missing,
                     "missing_change": new_missing - ref["missing_rate"]})
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values("psi", ascending=False, na_position="first").reset_index(drop=True)
    return out


def score_drift(reference: dict, pd_new: np.ndarray, thresholds: DriftThresholds | None = None) -> dict | None:
    t = thresholds or DriftThresholds()
    ref = reference.get("score")
    if not ref:
        return None
    props = _numeric_props(pd.Series(np.asarray(pd_new, dtype=float)), ref["cuts"])
    value = psi_from_props(np.asarray(ref["props"]), props)
    return {"psi": value, "status": psi_status(value, t.psi_moderate, t.psi_significant), "ref_mean_pd": ref["mean"],
            "new_mean_pd": float(np.mean(pd_new)), "ref_props": ref["props"], "new_props": props.tolist(),
            "cuts": ref["cuts"]}


@dataclass(frozen=True)
class DriftThresholds:
    """Alert thresholds. Defaults are the common industry rules of thumb."""

    psi_moderate: float = defaults.PSI_MODERATE
    psi_significant: float = defaults.PSI_SIGNIFICANT
    score_psi: float = defaults.PSI_MODERATE
    missing_change: float = 0.10


@dataclass(frozen=True)
class DriftAlert:
    code: str  # missing_columns | type_changes | feature_shift | missingness_jump | score_shift
    severity: str  # "warning" | "critical"
    message: str
    value: float | None = None
    threshold: float | None = None
    columns: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {"code": self.code, "severity": self.severity, "message": self.message, "value": self.value,
                "threshold": self.threshold, "columns": list(self.columns)}


def drift_alert_records(feature_table: pd.DataFrame, score: dict | None, schema: dict,
                        thresholds: DriftThresholds | None = None) -> list[DriftAlert]:
    """Structured alerts for the monitoring service (same rules as :func:`drift_alerts`)."""
    t = thresholds or DriftThresholds()
    alerts: list[DriftAlert] = []
    if schema["missing_columns"]:
        cols = tuple(schema["missing_columns"])
        alerts.append(DriftAlert("missing_columns", "critical",
                                 f"{len(cols)} model input column(s) are missing from the new data.",
                                 float(len(cols)), 0.0, cols))
    if schema["type_changes"]:
        cols = tuple(c["column"] for c in schema["type_changes"])
        alerts.append(DriftAlert("type_changes", "critical",
                                 f"{len(cols)} column(s) changed type versus training.", float(len(cols)), 0.0, cols))
    if not feature_table.empty:
        psi = feature_table["psi"].astype(float)
        sig = feature_table[psi >= t.psi_significant]
        if len(sig):
            alerts.append(DriftAlert("feature_shift", "warning",
                                     f"{len(sig)} feature(s) show a significant distribution shift "
                                     f"(PSI ≥ {t.psi_significant}).",
                                     float(sig["psi"].max()), t.psi_significant, tuple(sig["column"])))
        jumps = feature_table[feature_table["missing_change"].astype(float).fillna(0).abs() >= t.missing_change]
        if len(jumps):
            alerts.append(DriftAlert("missingness_jump", "warning",
                                     f"{len(jumps)} feature(s) changed missingness by ≥ "
                                     f"{t.missing_change * 100:.0f} percentage points.",
                                     float(jumps["missing_change"].abs().max()), t.missing_change,
                                     tuple(jumps["column"])))
    if score and score["psi"] >= t.score_psi:
        alerts.append(DriftAlert("score_shift", "warning",
                                 f"The predicted-PD distribution has shifted (PSI = {score['psi']:.3f}).",
                                 float(score["psi"]), t.score_psi))
    return alerts


def drift_alerts(feature_table: pd.DataFrame, score: dict | None, schema: dict,
                 thresholds: DriftThresholds | None = None) -> list[str]:
    """Human-readable alert messages (v1 interface)."""
    return [a.message for a in drift_alert_records(feature_table, score, schema, thresholds)]
