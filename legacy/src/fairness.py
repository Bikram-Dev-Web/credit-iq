"""Group-level validation for a responsible-lending review.

Sensitive attributes are NOT model inputs by default. They are used here only
to audit outcomes, as a separately documented analysis. No single metric can
establish that a model is "fair" — different fairness criteria can be mutually
incompatible when base rates differ between groups.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .data_validation import KIND_NUMERIC, infer_kind

MIN_GROUP_SIZE = 30


def make_groups(series: pd.Series, column: str, max_groups: int = 8) -> pd.Series:
    """Turn a sensitive column into audit groups (age bands for DAYS_BIRTH/age, top-k levels otherwise)."""
    s = series
    if infer_kind(s) == KIND_NUMERIC:
        v = pd.to_numeric(s, errors="coerce")
        if "days_birth" in column.lower():
            v = -v / 365.25
        if "birth" in column.lower() or "age" in column.lower():
            bins = [0, 25, 35, 45, 55, 65, 200]
            labels = ["<25", "25–34", "35–44", "45–54", "55–64", "65+"]
            return pd.cut(v, bins=bins, labels=labels, right=False).astype(object).where(v.notna(), "Missing").astype(str)
        try:
            return pd.qcut(v, 4, duplicates="drop").astype(str).where(v.notna(), "Missing")
        except ValueError:
            return v.astype(str)
    key = s.astype(object).where(s.notna(), "Missing").astype(str)
    counts = key.value_counts()
    if len(counts) > max_groups:
        key = key.where(key.isin(counts.index[:max_groups]), "Other")
    return key


def group_metrics(y, pd_, threshold: float, groups: pd.Series, min_group: int = MIN_GROUP_SIZE) -> pd.DataFrame:
    y = np.asarray(y).astype(int)
    p = np.asarray(pd_, dtype=float)
    g = np.asarray(groups).astype(str)
    flagged = p >= threshold
    rows = []
    for grp in pd.unique(g):
        m = g == grp
        yy, pp, ff = y[m], p[m], flagged[m]
        n = int(m.sum())
        pos, neg = int(yy.sum()), int(n - yy.sum())
        tp = int((ff & (yy == 1)).sum())
        fp = int((ff & (yy == 0)).sum())
        auc = roc_auc_score(yy, pp) if pos and neg else None
        rows.append({
            "group": grp, "n": n, "share": n / len(y),
            "observed_default_rate": yy.mean() if n else None,
            "mean_pd": pp.mean() if n else None,
            "calibration_gap": (pp.mean() - yy.mean()) if n else None,
            "flag_rate": ff.mean() if n else None,
            "within_threshold_rate": 1 - ff.mean() if n else None,
            "tpr_recall": tp / pos if pos else None,
            "fnr_missed_defaults": 1 - tp / pos if pos else None,
            "fpr_good_flagged": fp / neg if neg else None,
            "precision": tp / (tp + fp) if (tp + fp) else None,
            "roc_auc": auc,
            "small_sample": n < min_group or pos < 10,
        })
    return pd.DataFrame(rows).sort_values("n", ascending=False).reset_index(drop=True)


def disparity_summary(table: pd.DataFrame) -> pd.DataFrame:
    """Gaps between the most and least favourable groups (groups with small samples excluded)."""
    t = table[~table["small_sample"]]
    if len(t) < 2:
        return pd.DataFrame()
    rows = []

    def add(metric: str, label: str, kind: str, ratio_ok: bool = True):
        vals = t[["group", metric]].dropna()
        if len(vals) < 2:
            return
        hi = vals.loc[vals[metric].idxmax()]
        lo = vals.loc[vals[metric].idxmin()]
        ratio = lo[metric] / hi[metric] if ratio_ok and hi[metric] else None
        rows.append({"metric": label, "highest_group": hi["group"], "highest": hi[metric],
                     "lowest_group": lo["group"], "lowest": lo[metric], "difference": hi[metric] - lo[metric],
                     "ratio_low_to_high": ratio, "what_it_checks": kind})

    add("within_threshold_rate", "Share below the PD threshold (1 − flag rate)",
        "Demographic parity / adverse-impact ratio (the US 'four-fifths' rule of thumb compares this ratio with 0.8)")
    add("tpr_recall", "Recall of defaulters (TPR)", "Equal opportunity — are defaulters caught equally across groups?")
    add("fpr_good_flagged", "Good applicants flagged (FPR)", "Are creditworthy applicants burdened equally?")
    add("calibration_gap", "Calibration gap (mean PD − observed)", "Does a given PD mean the same risk in each group?", ratio_ok=False)
    add("roc_auc", "ROC-AUC within group", "Is ranking quality similar across groups?")
    return pd.DataFrame(rows)


def cramers_v(x: pd.Series, y: pd.Series) -> float | None:
    ct = pd.crosstab(x, y)
    if ct.shape[0] < 2 or ct.shape[1] < 2:
        return None
    n = ct.to_numpy().sum()
    expected = np.outer(ct.sum(axis=1), ct.sum(axis=0)) / n
    chi2 = float(((ct.to_numpy() - expected) ** 2 / np.where(expected == 0, 1, expected)).sum())
    k = min(ct.shape) - 1
    return float(np.sqrt(chi2 / (n * k))) if k else None


def correlation_ratio(groups: pd.Series, values: pd.Series) -> float | None:
    """η: share of a numeric variable's spread explained by group membership (0–1)."""
    v = pd.to_numeric(values, errors="coerce")
    m = v.notna()
    v, g = v[m], groups[m]
    if len(v) < 10 or v.var() == 0:
        return None
    grand = v.mean()
    between = sum(len(vals) * (vals.mean() - grand) ** 2 for _, vals in v.groupby(g))
    total = ((v - grand) ** 2).sum()
    return float(np.sqrt(between / total)) if total else None


def proxy_scan(df: pd.DataFrame, groups: pd.Series, predictors: list[str], sample: int = 15_000, seed: int = 42) -> pd.DataFrame:
    """Association between each model input and the audit groups (possible proxy variables)."""
    data = df[predictors].copy()
    data["__g__"] = np.asarray(groups)
    if len(data) > sample:
        data = data.sample(sample, random_state=seed)
    g = data.pop("__g__").astype(str)
    rows = []
    for col in predictors:
        s = data[col]
        if infer_kind(s) == KIND_NUMERIC:
            val, method = correlation_ratio(g, s), "correlation ratio (η)"
        else:
            val, method = cramers_v(s.astype(str).fillna("Missing"), g), "Cramér's V"
        level = None if val is None else ("strong" if val >= 0.3 else "moderate" if val >= 0.15 else "weak")
        rows.append({"feature": col, "association": val, "method": method, "level": level})
    return pd.DataFrame(rows).sort_values("association", ascending=False, na_position="last").reset_index(drop=True)


def missingness_by_group(df: pd.DataFrame, groups: pd.Series, columns: list[str], top: int = 10) -> pd.DataFrame:
    miss = df[columns].isna()
    miss["__g__"] = np.asarray(groups)
    by = miss.groupby("__g__").mean().T
    if by.shape[1] < 2:
        return pd.DataFrame()
    by["max_gap"] = by.max(axis=1) - by.min(axis=1)
    return by.sort_values("max_gap", ascending=False).head(top).reset_index().rename(columns={"index": "feature"})
