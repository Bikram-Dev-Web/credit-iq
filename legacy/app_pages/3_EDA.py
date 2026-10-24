"""Page 3 — Exploratory Data Analysis (historical observations only, no model)."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import data_validation as dv
from src import ui
from src.portfolio_analysis import quantile_bands

ss = st.session_state
ui.page_header("Exploratory Analysis", "How the observed default rate varies across borrower and loan characteristics. "
               "Everything here is a historical observation, not a prediction.")
ui.require_target()

MAX_ROWS = 150_000
t = ss.target
df_all = ss.df
y_all = dv.apply_target_mapping(df_all[t.column], t.positive_values, t.negative_values)
labelled = df_all.loc[y_all.notna()]
y = y_all.loc[y_all.notna()].astype(int)
if len(labelled) > MAX_ROWS:
    idx = labelled.sample(MAX_ROWS, random_state=42).index
    labelled, y = labelled.loc[idx], y.loc[idx]
    ui.note(f"Charts use a random sample of {MAX_ROWS:,} labelled rows.")
base_rate = float(y.mean())
features = ss.predictors or [c for c in labelled.columns if c != t.column]
all_cols = [c for c in labelled.columns if c != t.column]


def _assoc():
    key = (ss.ds_info.fingerprint, t.column, tuple(features))
    cache = ss.setdefault("_eda_assoc", {})
    if key not in cache:
        cache.clear()
        cache[key] = dv.leakage_scan(labelled, y, features, sample_size=15_000)
    return cache[key]


def _rate_by(key: pd.Series) -> pd.DataFrame:
    g = pd.DataFrame({"k": key.values, "y": y.values}).groupby("k", dropna=False, observed=False)
    out = pd.DataFrame({"rows": g.size(), "defaults": g["y"].sum(), "default_rate": g["y"].mean()}).reset_index()
    return out


# ---------------------------------------------------------------- Key findings
assoc = _assoc()
findings = [f"Observed default rate: <b>{ui.pct(base_rate, 2)}</b> ({int(y.sum()):,} of {len(y):,} labelled applications)."]
top = assoc.dropna(subset=["single_feature_auc"]).head(3)
if len(top):
    findings.append("Strongest single-column signals (direction-free AUC): " +
                    ", ".join(f"<b>{r.column}</b> ({r.single_feature_auc:.2f})" for r in top.itertuples()) + ".")
miss = labelled[features].isna().mean().sort_values(ascending=False)
if len(miss) and miss.iloc[0] > 0:
    findings.append(f"Most incomplete predictor: <b>{miss.index[0]}</b> ({ui.pct(miss.iloc[0])} missing).")
best = None
for col in features:
    if dv.infer_kind(labelled[col]) == dv.KIND_CATEGORICAL and labelled[col].nunique() <= 60:
        r = _rate_by(labelled[col].astype(str))
        r = r[r["rows"] >= max(100, int(0.005 * len(y)))]
        if len(r):
            hi = r.loc[r["default_rate"].idxmax()]
            if best is None or hi["default_rate"] > best[2]:
                best = (col, hi["k"], hi["default_rate"], int(hi["rows"]))
if best:
    findings.append(f"Highest-risk sizeable category: <b>{best[0]} = {best[1]}</b> with {ui.pct(best[2])} default rate "
                    f"({best[3]:,} applications), vs {ui.pct(base_rate)} overall.")
ui.callout("<br>".join(findings))
ui.note("Associations describe this historical sample. They are not causal, and a strong association can still reflect "
        "how the data were collected.")

tab_out, tab_feat, tab_miss, tab_seg, tab_assoc = ui.tabs(["Outcome", "Feature explorer", "Missing values", "Segments", "Associations"], key="tabs_eda")

with tab_out:
    c1, c2 = st.columns([1, 1.6])
    with c1:
        fig = go.Figure(go.Bar(x=["Non-default", "Default"], y=[int((y == 0).sum()), int(y.sum())],
                               marker_color=[ui.REPAID_COLOR, ui.DEFAULT_COLOR],
                               text=[ui.pct(1 - base_rate), ui.pct(base_rate)], textposition="outside",
                               hovertemplate="%{x}: %{y:,} applications<extra></extra>"))
        fig.update_layout(title="Observed outcomes", yaxis_title="Applications")
        ui.plot(fig)
    with c2:
        if ss.date_col and ss.date_col in labelled.columns:
            d = dv.parse_dates(labelled[ss.date_col])
            tmp = pd.DataFrame({"period": d.dt.to_period("M").astype(str), "y": y.values}).loc[d.notna().values]
            g = tmp.groupby("period")["y"].agg(["mean", "size"]).reset_index()
            fig = go.Figure()
            fig.add_bar(x=g["period"], y=g["size"], name="Applications", marker_color="#D5DEEA", yaxis="y2",
                        hovertemplate="%{x}: %{y:,} applications<extra></extra>")
            fig.add_scatter(x=g["period"], y=g["mean"], name="Default rate", mode="lines+markers", line_color=ui.DEFAULT_COLOR,
                            hovertemplate="%{x}: %{y:.2%}<extra></extra>")
            fig.update_layout(title="Default rate by application month", yaxis=dict(tickformat=".0%", title="Default rate"),
                              yaxis2=dict(overlaying="y", side="right", showgrid=False, title="Applications"))
            ui.plot(fig)
        else:
            ui.empty_state("No date column", "Select an application date column in the Data Center to see time trends.")

with tab_feat:
    col = st.selectbox("Feature", all_cols, index=all_cols.index(features[0]) if features and features[0] in all_cols else 0)
    s = labelled[col]
    kind = dv.infer_kind(s)
    if kind == dv.KIND_NUMERIC and s.nunique() > 12:
        v = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)
        lo, hi = v.quantile([0.01, 0.99])
        clip = st.checkbox("Clip to 1st–99th percentile for display", value=True)
        vv = v.clip(lo, hi) if clip else v
        c1, c2 = st.columns(2)
        with c1:
            fig = go.Figure()
            fig.add_histogram(x=vv[y == 0], name="Non-default", marker_color=ui.REPAID_COLOR, histnorm="percent", opacity=0.75, nbinsx=40)
            fig.add_histogram(x=vv[y == 1], name="Default", marker_color=ui.DEFAULT_COLOR, histnorm="percent", opacity=0.75, nbinsx=40)
            fig.update_layout(barmode="overlay", title=f"Distribution of {col} by outcome", yaxis_title="% of group", xaxis_title=col)
            ui.plot(fig)
        with c2:
            bands = quantile_bands(v, q=10, label_fmt="{:,.4g}")
            r = _rate_by(bands)
            r = r[r["rows"] > 0]
            fig = go.Figure(go.Bar(x=r["k"].astype(str), y=r["default_rate"], marker_color="#1E4C8A",
                                   customdata=r[["rows"]], hovertemplate="%{x}<br>Default rate %{y:.2%}<br>%{customdata[0]:,} rows<extra></extra>"))
            fig.add_hline(y=base_rate, line_dash="dot", line_color=ui.STEEL, annotation_text="overall", annotation_position="top left")
            fig.update_layout(title=f"Default rate by {col} decile", yaxis_tickformat=".0%", xaxis_title=None)
            ui.plot(fig)
        stats = pd.DataFrame({"Non-default": v[y == 0].describe(), "Default": v[y == 1].describe()}).T.reset_index().rename(columns={"index": "group"})
        ui.show_table(stats, dec_cols=[c for c in stats.columns if c not in ("group", "count")], int_cols=["count"])
    else:
        key = s.astype(object).where(s.notna(), "Missing").astype(str)
        r = _rate_by(key).sort_values("rows", ascending=False).head(25)
        fig = go.Figure()
        fig.add_bar(x=r["k"], y=r["rows"], name="Applications", marker_color="#D5DEEA", yaxis="y2",
                    hovertemplate="%{x}: %{y:,} rows<extra></extra>")
        fig.add_scatter(x=r["k"], y=r["default_rate"], name="Default rate", mode="markers+lines", line_color=ui.DEFAULT_COLOR,
                        hovertemplate="%{x}: %{y:.2%}<extra></extra>")
        fig.add_hline(y=base_rate, line_dash="dot", line_color=ui.STEEL)
        fig.update_layout(title=f"Default rate and volume by {col} (top 25 categories)", yaxis=dict(tickformat=".0%", title="Default rate"),
                          yaxis2=dict(overlaying="y", side="right", showgrid=False, title="Applications"))
        ui.plot(fig, height=420)
        ui.show_table(r.rename(columns={"k": col}), pct_cols=["default_rate"], int_cols=["rows", "defaults"], height=300)

with tab_miss:
    m = labelled[all_cols].isna().mean().sort_values(ascending=False)
    m = m[m > 0]
    if m.empty:
        ui.note("No missing values.")
    else:
        rows = []
        for c in m.index[:40]:
            isna = labelled[c].isna().values
            rows.append({"column": c, "missing_pct": m[c], "default_rate_when_missing": y.values[isna].mean(),
                         "default_rate_when_present": y.values[~isna].mean() if (~isna).any() else np.nan})
        mt = pd.DataFrame(rows)
        fig = go.Figure()
        fig.add_bar(y=mt["column"], x=mt["default_rate_when_missing"], name="Missing", orientation="h", marker_color=ui.DEFAULT_COLOR)
        fig.add_bar(y=mt["column"], x=mt["default_rate_when_present"], name="Present", orientation="h", marker_color=ui.REPAID_COLOR)
        fig.update_layout(barmode="group", title="Default rate when a value is missing vs present", xaxis_tickformat=".0%",
                          yaxis=dict(autorange="reversed"))
        ui.plot(fig, height=max(320, 26 * len(mt) + 90))
        ui.note("When missingness itself predicts default, the model's missing-value indicators capture it — but it can also "
                "signal a data-collection process that differs across groups.")
        ui.show_table(mt, pct_cols=["missing_pct", "default_rate_when_missing", "default_rate_when_present"])

with tab_seg:
    c1, c2, c3 = st.columns(3)
    a = c1.selectbox("Rows", all_cols, index=0, key="seg_a")
    b = c2.selectbox("Columns", all_cols, index=min(1, len(all_cols) - 1), key="seg_b")
    min_n = c3.number_input("Minimum applications per cell", 1, 10_000, 50)

    def _key(series):
        if dv.infer_kind(series) == dv.KIND_NUMERIC and series.nunique() > 12:
            return quantile_bands(series, q=5, label_fmt="{:,.4g}")
        k = series.astype(object).where(series.notna(), "Missing").astype(str)
        top_levels = k.value_counts().index[:12]
        return k.where(k.isin(top_levels), "Other")

    if a == b:
        st.info("Choose two different columns.")
    else:
        tmp = pd.DataFrame({"a": _key(labelled[a]).values, "b": _key(labelled[b]).values, "y": y.values})
        g = tmp.groupby(["a", "b"], observed=True)["y"].agg(["mean", "size"]).reset_index()
        g.loc[g["size"] < min_n, "mean"] = np.nan
        piv = g.pivot(index="a", columns="b", values="mean")
        cnt = g.pivot(index="a", columns="b", values="size")
        fig = go.Figure(go.Heatmap(z=piv.values, x=[str(c) for c in piv.columns], y=[str(i) for i in piv.index],
                                   customdata=cnt.reindex_like(piv).values, colorscale=[[0, "#EEF3F9"], [0.5, "#C9783A"], [1, "#A23B3B"]],
                                   colorbar=dict(tickformat=".0%", title="Default rate"),
                                   hovertemplate=f"{a}: %{{y}}<br>{b}: %{{x}}<br>Default rate %{{z:.2%}}<br>%{{customdata:,}} rows<extra></extra>"))
        fig.update_layout(title=f"Default rate by {a} × {b}", xaxis_title=b, yaxis_title=a)
        ui.plot(fig, height=460)
        ui.note(f"Blank cells have fewer than {min_n} applications.")

with tab_assoc:
    a = assoc.dropna(subset=["single_feature_auc"]).head(25)
    fig = go.Figure(go.Bar(x=a["single_feature_auc"], y=a["column"], orientation="h", marker_color="#1E4C8A",
                           hovertemplate="%{y}: AUC %{x:.3f}<extra></extra>"))
    fig.add_vline(x=0.5, line_dash="dot", line_color=ui.STEEL)
    fig.update_layout(title="Single-column predictive strength (direction-free AUC)", xaxis=dict(range=[0.45, 1.0]),
                      yaxis=dict(autorange="reversed"))
    ui.plot(fig, height=max(320, 22 * len(a) + 90))
    numeric = [c for c in a["column"] if dv.infer_kind(labelled[c]) == dv.KIND_NUMERIC][:15]
    if len(numeric) >= 2:
        corr = labelled[numeric].apply(pd.to_numeric, errors="coerce").corr(method="spearman")
        fig = go.Figure(go.Heatmap(z=corr.values, x=numeric, y=numeric, zmin=-1, zmax=1,
                                   colorscale=[[0, "#1E4C8A"], [0.5, "#FFFFFF"], [1, "#A23B3B"]],
                                   hovertemplate="%{y} × %{x}: ρ = %{z:.2f}<extra></extra>"))
        fig.update_layout(title="Spearman correlation among the strongest numeric predictors")
        ui.plot(fig, height=520)
        ui.note("Highly correlated predictors share credit in explanations, so individual attributions can be unstable.")
