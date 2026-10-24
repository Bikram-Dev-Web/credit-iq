"""Portfolio exposure, expected loss, segmentation and concentration analytics.

Three different quantities are kept apart throughout:

* **Observed default rate** – share of loans whose historical label is default.
* **Average predicted PD** – mean of the model's calibrated probabilities.
* **Expected credit loss** – Σ PD × LGD × EAD under the user's LGD/EAD assumptions.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config
from .data_validation import KIND_NUMERIC, infer_kind, parse_dates
from .financial_calculations import expected_loss


def assign_risk_band(pd_, cutoffs=config.DEFAULT_BAND_CUTOFFS, labels=config.BAND_LABELS) -> pd.Series:
    cut = list(cutoffs)
    if sorted(cut) != cut or any(c <= 0 or c >= 1 for c in cut):
        raise ValueError("Risk-band cutoffs must be increasing and strictly between 0 and 1.")
    if len(labels) != len(cut) + 1:
        raise ValueError("Number of labels must be one more than the number of cutoffs.")
    bins = [-np.inf, *cut, np.inf]
    return pd.cut(pd.Series(np.asarray(pd_, dtype=float)), bins=bins, labels=list(labels), right=False)


def band_description(cutoffs=config.DEFAULT_BAND_CUTOFFS, labels=config.BAND_LABELS) -> list[str]:
    edges = [0.0, *cutoffs, 1.0]
    return [f"{lab}: {edges[i]:.1%} ≤ PD < {edges[i + 1]:.1%}" if i < len(labels) - 1 else f"{lab}: PD ≥ {edges[i]:.1%}"
            for i, lab in enumerate(labels)]


def build_portfolio(frame: pd.DataFrame, y_observed, pd_, exposure_col: str | None, lgd: float, ccf: float = 1.0,
                    default_exposure: float = 100_000.0, cutoffs=config.DEFAULT_BAND_CUTOFFS) -> pd.DataFrame:
    """Loan-level table with PD, EAD, LGD, EL and risk band appended to the original columns."""
    port = frame.reset_index(drop=True).copy()
    port["pd"] = np.asarray(pd_, dtype=float)
    port["observed_default"] = np.asarray(y_observed).astype(int) if y_observed is not None else np.nan
    if exposure_col and exposure_col in port.columns:
        exposure = pd.to_numeric(port[exposure_col], errors="coerce")
        fill = float(exposure.median()) if exposure.notna().any() else default_exposure
        port["exposure_imputed"] = exposure.isna()
        exposure = exposure.fillna(fill).clip(lower=0)
    else:
        exposure = pd.Series(default_exposure, index=port.index, dtype=float)
        port["exposure_imputed"] = True
    port["ead"] = exposure.astype(float) * ccf
    port["lgd"] = float(lgd)
    port["expected_loss"] = expected_loss(port["pd"].to_numpy(), port["lgd"].to_numpy(), port["ead"].to_numpy())
    port["risk_band"] = assign_risk_band(port["pd"], cutoffs).values
    return port


def portfolio_kpis(port: pd.DataFrame, high_risk_labels=("Elevated", "High")) -> dict:
    n = len(port)
    if n == 0:
        return {"n_loans": 0}
    ead = port["ead"].to_numpy()
    total_ead = float(ead.sum())
    obs = port["observed_default"]
    has_obs = obs.notna().all()
    high = port["risk_band"].isin(high_risk_labels).to_numpy()
    return {
        "n_loans": n,
        "total_ead": total_ead,
        "observed_default_rate": float(obs.mean()) if has_obs else None,
        "observed_default_rate_ead_weighted": float((obs * ead).sum() / total_ead) if has_obs and total_ead else None,
        "avg_pd": float(port["pd"].mean()),
        "ead_weighted_pd": float((port["pd"] * ead).sum() / total_ead) if total_ead else None,
        "total_el": float(port["expected_loss"].sum()),
        "el_rate": float(port["expected_loss"].sum() / total_ead) if total_ead else None,
        "high_risk_share_count": float(high.mean()),
        "high_risk_share_ead": float(ead[high].sum() / total_ead) if total_ead else None,
    }


def summarise(port: pd.DataFrame, key) -> pd.DataFrame:
    total_ead = port["ead"].sum()
    total_el = port["expected_loss"].sum()
    g = port.groupby(key, observed=False, dropna=False)
    out = pd.DataFrame({
        "loans": g.size(),
        "ead": g["ead"].sum(),
        "observed_default_rate": g["observed_default"].mean(),
        "avg_pd": g["pd"].mean(),
        "expected_loss": g["expected_loss"].sum(),
    })
    out["loan_share"] = out["loans"] / max(len(port), 1)
    out["ead_share"] = out["ead"] / total_ead if total_ead else np.nan
    out["el_share"] = out["expected_loss"] / total_el if total_el else np.nan
    out["el_rate"] = out["expected_loss"] / out["ead"].replace(0, np.nan)
    return out.reset_index()


def band_summary(port: pd.DataFrame) -> pd.DataFrame:
    return summarise(port, "risk_band")


def quantile_bands(series: pd.Series, q: int = 5, label_fmt: str = "{:,.0f}") -> pd.Series:
    """Quantile bands as an ORDERED categorical (so charts sort low → high), with a 'Missing' level if needed."""
    v = pd.to_numeric(series, errors="coerce")
    try:
        cats = pd.qcut(v, q=q, duplicates="drop")
    except (ValueError, IndexError):
        return pd.Series(pd.Categorical(["All"] * len(v)), index=v.index)
    labels, seen = [], set()
    for i, iv in enumerate(cats.cat.categories):
        lab = f"{label_fmt.format(iv.left)} – {label_fmt.format(iv.right)}"
        if lab in seen:
            lab = f"{lab} ({i + 1})"
        seen.add(lab)
        labels.append(lab)
    out = cats.cat.rename_categories(labels)
    if v.isna().any():
        out = out.cat.add_categories(["Missing"]).fillna("Missing")
    return out


def segment_summary(port: pd.DataFrame, column: str, top: int = 15) -> pd.DataFrame:
    if column not in port.columns:
        raise KeyError(column)
    s = port[column]
    if infer_kind(s) == KIND_NUMERIC and s.nunique() > 12:
        key = quantile_bands(s)
    else:
        key = s.astype(object).where(s.notna(), "Missing").astype(str)
        counts = key.value_counts()
        if len(counts) > top:
            keep = set(counts.index[:top])
            key = key.where(key.isin(keep), "Other (grouped)")
    tmp = port.assign(_segment=key.values)
    out = summarise(tmp, "_segment").rename(columns={"_segment": column})
    out = out[out["loans"] > 0]
    if isinstance(key.dtype, pd.CategoricalDtype):
        return out.reset_index(drop=True)  # numeric bands keep their natural order
    return out.sort_values("ead", ascending=False).reset_index(drop=True)


def concentration_metrics(summary: pd.DataFrame) -> dict:
    """Herfindahl–Hirschman index of exposure shares and top-segment shares."""
    shares = summary["ead_share"].fillna(0).sort_values(ascending=False).to_numpy()
    if shares.size == 0:
        return {"hhi": None, "top1_share": None, "top3_share": None, "effective_segments": None}
    hhi = float(np.sum(shares ** 2))
    return {"hhi": hhi, "top1_share": float(shares[:1].sum()), "top3_share": float(shares[:3].sum()),
            "effective_segments": 1 / hhi if hhi else None}


def loss_concentration_curve(port: pd.DataFrame, points: int = 101) -> pd.DataFrame:
    el = np.sort(port["expected_loss"].to_numpy())[::-1]
    if el.sum() <= 0:
        return pd.DataFrame({"share_of_loans": [0, 1], "share_of_expected_loss": [0, 1]})
    cum = np.concatenate([[0], np.cumsum(el)]) / el.sum()
    x = np.linspace(0, 1, len(cum))
    grid = np.linspace(0, 1, points)
    return pd.DataFrame({"share_of_loans": grid, "share_of_expected_loss": np.interp(grid, x, cum)})


def pd_decile_view(port: pd.DataFrame, groups: int = 10) -> pd.DataFrame:
    """Average predicted PD vs observed default rate per PD decile (1 = lowest risk)."""
    if len(port) < groups:
        return pd.DataFrame()
    ranks = port["pd"].rank(method="first")
    dec = pd.qcut(ranks, groups, labels=range(1, groups + 1))
    g = port.assign(decile=dec.astype(int)).groupby("decile")
    return pd.DataFrame({"loans": g.size(), "avg_pd": g["pd"].mean(),
                         "observed_default_rate": g["observed_default"].mean(),
                         "ead": g["ead"].sum(), "expected_loss": g["expected_loss"].sum()}).reset_index()


def time_trend(port: pd.DataFrame, date_col: str, freq: str = "M") -> pd.DataFrame:
    d = parse_dates(port[date_col])
    tmp = port.assign(_period=d.dt.to_period(freq).astype(str)).loc[d.notna()]
    if tmp.empty:
        return pd.DataFrame()
    g = tmp.groupby("_period")
    return pd.DataFrame({"loans": g.size(), "observed_default_rate": g["observed_default"].mean(),
                         "avg_pd": g["pd"].mean(), "ead": g["ead"].sum(),
                         "expected_loss": g["expected_loss"].sum()}).reset_index().rename(columns={"_period": "period"})


def filter_portfolio(port: pd.DataFrame, cat_filters: dict | None = None, ead_range: tuple | None = None,
                     date_col: str | None = None, date_range: tuple | None = None) -> pd.DataFrame:
    mask = pd.Series(True, index=port.index)
    for col, values in (cat_filters or {}).items():
        if values:
            mask &= port[col].astype(str).isin([str(v) for v in values])
    if ead_range:
        mask &= port["ead"].between(ead_range[0], ead_range[1])
    if date_col and date_range and date_col in port.columns:
        d = parse_dates(port[date_col])
        mask &= d.between(pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1]))
    return port.loc[mask]
