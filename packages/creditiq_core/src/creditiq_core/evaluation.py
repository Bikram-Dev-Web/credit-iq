"""Model evaluation: discrimination, calibration, threshold and lift diagnostics.

Undefined metrics (e.g. ROC-AUC with a single class, precision with no flagged
applicants) are returned as ``None`` rather than a misleading number.
"""
from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

METRIC_HELP = {
    "roc_auc": "Probability that a randomly chosen defaulter is scored riskier than a randomly chosen non-defaulter. 0.5 = random ranking.",
    "gini": "Gini = 2×AUC − 1, the ranking measure most often quoted in credit scoring.",
    "pr_auc": "Average precision: how well the model concentrates defaulters at the top of the risk ranking. Compare with the default rate (its random baseline).",
    "precision": "Of the applicants flagged, the share who actually defaulted.",
    "recall": "Of the actual defaulters, the share the model flagged (sensitivity). Missed defaulters become credit losses.",
    "specificity": "Of the non-defaulters, the share NOT flagged. Low specificity means good customers are turned away or delayed.",
    "f1": "Harmonic mean of precision and recall at the chosen threshold.",
    "balanced_accuracy": "Average of recall and specificity — accuracy that is not dominated by the majority (non-default) class.",
    "brier": "Mean squared error of the PD against the 0/1 outcome. Lower is better; rewards calibrated probabilities.",
    "log_loss": "Penalty for confident wrong probabilities. Lower is better.",
    "ks": "Kolmogorov–Smirnov: maximum gap between the score distributions of defaulters and non-defaulters.",
    "prevalence": "Observed default rate in the evaluated records (historical labels, not a prediction).",
    "lift_top10": "Default rate among the riskiest 10% divided by the overall default rate.",
    "ece": "Expected calibration error: average gap between predicted PD and observed default rate across bins.",
    "flag_rate": "Share of applicants whose PD is at or above the threshold (flagged for review).",
}


def _safe(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> float | None:
    try:
        val = fn(*args, **kwargs)
        return None if val is None or (isinstance(val, float) and not math.isfinite(val)) else float(val)
    except ValueError:
        return None


def _div(a: float, b: float) -> float | None:
    return float(a) / float(b) if b else None


def confusion_counts(y_true: ArrayLike, y_prob: ArrayLike, threshold: float) -> dict[str, int]:
    y = np.asarray(y_true).astype(int)
    pred = (np.asarray(y_prob, dtype=float) >= threshold).astype(int)
    return {
        "tp": int(((pred == 1) & (y == 1)).sum()),
        "fp": int(((pred == 1) & (y == 0)).sum()),
        "tn": int(((pred == 0) & (y == 0)).sum()),
        "fn": int(((pred == 0) & (y == 1)).sum()),
    }


def ks_statistic(y_true: ArrayLike, y_prob: ArrayLike) -> tuple[float | None, float | None]:
    y = np.asarray(y_true).astype(int)
    if len(np.unique(y)) < 2:
        return None, None
    fpr, tpr, thr = roc_curve(y, y_prob)
    i = int(np.argmax(tpr - fpr))
    return float(tpr[i] - fpr[i]), float(thr[i]) if np.isfinite(thr[i]) else None


def lift_at(y_true: ArrayLike, y_prob: ArrayLike, fraction: float = 0.10) -> float | None:
    y = np.asarray(y_true).astype(int)
    if len(y) == 0 or y.mean() == 0:
        return None
    k = max(int(round(len(y) * fraction)), 1)
    top = np.argsort(-np.asarray(y_prob, dtype=float), kind="stable")[:k]
    return float(y[top].mean() / y.mean())


def classification_metrics(y_true: ArrayLike, y_prob: ArrayLike, threshold: float = 0.5) -> dict[str, Any]:
    """Full metric set at one threshold. Values are ``None`` when undefined."""
    y = np.asarray(y_true).astype(int)
    p = np.asarray(y_prob, dtype=float)
    if len(y) == 0:
        raise ValueError("Cannot evaluate an empty set.")
    if np.any((p < 0) | (p > 1)) or np.any(~np.isfinite(p)):
        raise ValueError("Probabilities must be finite and within [0, 1].")
    two_classes = len(np.unique(y)) == 2
    c = confusion_counts(y, p, threshold)
    tp, fp, tn, fn = c["tp"], c["fp"], c["tn"], c["fn"]
    precision = _div(tp, tp + fp)
    recall = _div(tp, tp + fn)
    specificity = _div(tn, tn + fp)
    f1 = None if precision is None or recall is None or (precision + recall) == 0 else 2 * precision * recall / (precision + recall)
    bal = None if recall is None or specificity is None else (recall + specificity) / 2
    roc = _safe(roc_auc_score, y, p) if two_classes else None
    ks, _ = ks_statistic(y, p)
    return {
        "n": int(len(y)),
        "prevalence": float(y.mean()),
        "mean_pd": float(p.mean()),
        "threshold": float(threshold),
        **c,
        "flagged": tp + fp,
        "flag_rate": (tp + fp) / len(y),
        "roc_auc": roc,
        "gini": None if roc is None else 2 * roc - 1,
        "pr_auc": _safe(average_precision_score, y, p) if two_classes else None,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "balanced_accuracy": bal,
        "accuracy": (tp + tn) / len(y),
        "brier": float(brier_score_loss(y, p)),
        "log_loss": _safe(log_loss, y, np.clip(p, 1e-12, 1 - 1e-12), labels=[0, 1]),
        "ks": ks,
        "lift_top10": lift_at(y, p, 0.10),
    }


def _downsample(*arrays: np.ndarray, max_points: int = 400) -> tuple[np.ndarray, ...]:
    n = len(arrays[0])
    if n <= max_points:
        return arrays
    idx = np.unique(np.linspace(0, n - 1, max_points).astype(int))
    return tuple(a[idx] for a in arrays)


def roc_points(y_true: ArrayLike, y_prob: ArrayLike) -> pd.DataFrame:
    y = np.asarray(y_true).astype(int)
    if len(np.unique(y)) < 2:
        return pd.DataFrame(columns=["fpr", "tpr", "threshold"])
    fpr, tpr, thr = roc_curve(y, y_prob)
    fpr, tpr, thr = _downsample(fpr, tpr, thr)
    return pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": np.clip(thr, 0, 1)})


def pr_points(y_true: ArrayLike, y_prob: ArrayLike) -> pd.DataFrame:
    y = np.asarray(y_true).astype(int)
    if len(np.unique(y)) < 2:
        return pd.DataFrame(columns=["recall", "precision", "threshold"])
    prec, rec, thr = precision_recall_curve(y, y_prob)
    thr = np.append(thr, 1.0)
    rec, prec, thr = _downsample(rec, prec, thr)
    return pd.DataFrame({"recall": rec, "precision": prec, "threshold": thr})


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float | None, float | None]:
    if n <= 0:
        return None, None
    phat = k / n
    denom = 1 + z ** 2 / n
    centre = (phat + z ** 2 / (2 * n)) / denom
    half = z * math.sqrt(phat * (1 - phat) / n + z ** 2 / (4 * n ** 2)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def calibration_table(y_true: ArrayLike, y_prob: ArrayLike, n_bins: int = 10, strategy: str = "quantile") -> pd.DataFrame:
    """Reliability table: mean predicted PD vs observed default rate per bin, with Wilson 95% CIs."""
    y = np.asarray(y_true).astype(int)
    p = np.asarray(y_prob, dtype=float)
    if len(y) == 0:
        return pd.DataFrame()
    if strategy == "quantile":
        edges = np.unique(np.quantile(p, np.linspace(0, 1, n_bins + 1)))
    else:
        edges = np.linspace(0, 1, n_bins + 1)
    if len(edges) < 2:
        edges = np.array([p.min(), p.max() + 1e-9])
    bins = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, len(edges) - 2)
    rows = []
    for b in range(len(edges) - 1):
        m = bins == b
        n = int(m.sum())
        if n == 0:
            continue
        k = int(y[m].sum())
        lo, hi = wilson_interval(k, n)
        rows.append({"bin": b + 1, "pd_from": float(edges[b]), "pd_to": float(edges[b + 1]), "n": n,
                     "defaults": k, "mean_pd": float(p[m].mean()), "observed_rate": k / n,
                     "ci_low": lo, "ci_high": hi})
    return pd.DataFrame(rows)


def expected_calibration_error(table: pd.DataFrame) -> float | None:
    if table is None or table.empty:
        return None
    w = table["n"] / table["n"].sum()
    return float((w * (table["mean_pd"] - table["observed_rate"]).abs()).sum())


def threshold_sweep(y_true: ArrayLike, y_prob: ArrayLike, thresholds: ArrayLike | None = None) -> pd.DataFrame:
    """Vectorised metrics across thresholds (for the interactive explorer)."""
    y = np.asarray(y_true).astype(int)
    p = np.asarray(y_prob, dtype=float)
    if thresholds is None:
        thresholds = np.round(np.linspace(0.01, 0.99, 99), 2)
    thresholds = np.asarray(thresholds, dtype=float)
    order = np.argsort(p)
    p_sorted, y_sorted = p[order], y[order]
    pos_total = int(y.sum())
    neg_total = len(y) - pos_total
    # number of records with p >= t
    idx = np.searchsorted(p_sorted, thresholds, side="left")
    cum_pos_below = np.concatenate([[0], np.cumsum(y_sorted)])[idx]
    flagged = len(y) - idx
    tp = pos_total - cum_pos_below
    fp = flagged - tp
    fn = pos_total - tp
    tn = neg_total - fp
    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where(flagged > 0, tp / np.maximum(flagged, 1), np.nan)
        recall = np.where(pos_total > 0, tp / max(pos_total, 1), np.nan)
        spec = np.where(neg_total > 0, tn / max(neg_total, 1), np.nan)
        f1 = np.where((precision + recall) > 0, 2 * precision * recall / (precision + recall), np.nan)
    return pd.DataFrame({
        "threshold": thresholds, "flagged": flagged, "flag_rate": flagged / max(len(y), 1),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "precision": precision, "recall": recall, "specificity": spec, "f1": f1,
        "balanced_accuracy": (recall + spec) / 2,
    })


THRESHOLD_RULES = {
    "max_f1": "Maximise F1 on the validation set",
    "youden": "Maximise recall + specificity − 1 (Youden's J) on the validation set",
    "prevalence": "Flag the same share of applicants as the validation default rate",
    "manual": "Manually chosen threshold",
}


def select_threshold(y_true: ArrayLike, y_prob: ArrayLike, rule: str = "max_f1", manual: float = 0.5) -> float:
    """Choose a decision threshold on VALIDATION data. Never call this with test data."""
    y = np.asarray(y_true).astype(int)
    p = np.asarray(y_prob, dtype=float)
    if rule == "manual":
        return float(manual)
    if rule == "prevalence":
        return float(np.quantile(p, 1 - y.mean())) if len(y) else 0.5
    grid = np.unique(np.round(np.quantile(p, np.linspace(0.01, 0.99, 199)), 4))
    sweep = threshold_sweep(y, p, grid)
    if rule == "youden":
        score = sweep["recall"] + sweep["specificity"] - 1
    else:
        score = sweep["f1"]
    if score.isna().all():
        return 0.5
    # first maximum, skipping undefined values (same as Series.idxmax on this RangeIndex)
    return float(sweep["threshold"].to_numpy()[int(np.nanargmax(score.to_numpy(dtype=float)))])


def decile_table(y_true: ArrayLike, y_prob: ArrayLike, n_groups: int = 10) -> pd.DataFrame:
    """Risk-ranked groups (1 = riskiest) with default rate, capture and lift."""
    y = np.asarray(y_true).astype(int)
    p = np.asarray(y_prob, dtype=float)
    if len(y) < n_groups:
        return pd.DataFrame()
    order = np.argsort(-p, kind="stable")
    groups = np.array_split(order, n_groups)
    total_def = max(int(y.sum()), 1)
    base = float(y.mean()) if y.mean() > 0 else np.nan
    rows, cum_def, cum_n = [], 0, 0
    for i, g in enumerate(groups, start=1):
        d = int(y[g].sum())
        cum_def += d
        cum_n += len(g)
        rows.append({
            "risk_group": i, "n": len(g), "min_pd": float(p[g].min()), "max_pd": float(p[g].max()),
            "mean_pd": float(p[g].mean()), "defaults": d, "default_rate": d / len(g),
            "lift": (d / len(g)) / base if base == base else np.nan,
            "cum_capture": cum_def / total_def, "cum_lift": (cum_def / cum_n) / base if base == base else np.nan,
        })
    return pd.DataFrame(rows)


def metrics_by_period(y_true: ArrayLike, y_prob: ArrayLike, dates: pd.Series, threshold: float, freq: str = "Q") -> pd.DataFrame:
    """Performance per calendar period (only when real dates exist)."""
    d = pd.to_datetime(pd.Series(dates).reset_index(drop=True), errors="coerce")
    frame = pd.DataFrame({"y": np.asarray(y_true).astype(int), "p": np.asarray(y_prob, dtype=float), "d": d}).dropna(subset=["d"])
    if frame.empty:
        return pd.DataFrame()
    frame["period"] = frame["d"].dt.to_period(freq).astype(str)
    rows = []
    for period, g in frame.groupby("period"):
        m = classification_metrics(g["y"], g["p"], threshold)
        rows.append({"period": period, "n": m["n"], "observed_default_rate": m["prevalence"],
                     "mean_pd": m["mean_pd"], "roc_auc": m["roc_auc"], "recall": m["recall"],
                     "precision": m["precision"], "brier": m["brier"]})
    return pd.DataFrame(rows)
