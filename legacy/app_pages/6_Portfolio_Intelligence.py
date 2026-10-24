"""Page 6 — Portfolio Intelligence: exposure, expected loss, risk bands, segments and concentration."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import data_validation as dv
from src import portfolio_analysis as pa
from src import state, ui

ss = st.session_state
ui.page_header("Portfolio Intelligence", "Where exposure and expected credit loss sit across risk bands, segments and time — "
               "kept separate from the observed default rate and from the model's average PD.")
ui.require_experiment("portfolio analysis")

exp = ss.experiment
key = ss.active_model_key or exp.selected
fin = state.fin()

# ---------------------------------------------------------------- Scope and assumptions
with st.container(border=True):
    c1, c2, c3, c4 = st.columns([1.6, 1, 1, 1])
    scope = ui.pick(c1.selectbox, "Loans analysed", {"test": "Held-out test set (out-of-sample)", "validation": "Validation set",
                                                    "all": "All sampled rows (includes training rows — optimistic)"})
    fin["lgd"] = ui.pct_input(c2, "LGD", fin["lgd"], step=1.0, digits=1, help="Loss given default: share of EAD lost after recoveries (assumption).")
    fin["ccf"] = c3.number_input("CCF", 0.0, 1.5, float(fin["ccf"]), 0.05, help="Credit conversion factor: EAD = exposure × CCF.")
    if exp.exposure_col:
        c4.text_input("Exposure column", exp.exposure_col, disabled=True)
    else:
        fin["default_exposure"] = c4.number_input("Exposure per loan", 0.0, value=float(fin["default_exposure"]), step=10_000.0,
                                                  help="No exposure column was selected, so every loan gets this exposure.")
    b1, b2, b3, b4 = st.columns([1, 1, 1, 1.6])
    cut = [b1.number_input("Moderate from (PD %)", 0.1, 99.0, ss.band_cutoffs[0] * 100, 0.5),
           b2.number_input("Elevated from (PD %)", 0.1, 99.0, ss.band_cutoffs[1] * 100, 0.5),
           b3.number_input("High from (PD %)", 0.1, 99.0, ss.band_cutoffs[2] * 100, 0.5)]
    cut = [c / 100 for c in cut]
    if sorted(cut) != cut or len(set(cut)) < 3:
        st.error("Band cutoffs must increase: Moderate < Elevated < High.")
        st.stop()
    ss.band_cutoffs = cut
    b4.caption("Risk bands: " + "; ".join(pa.band_description(cut)))

if scope == "all":
    order = np.concatenate([exp.idx(s) for s in ("train", "validation", "test")])
    frame, y = exp.frame.iloc[order], exp.y.to_numpy()[order]
    p = np.concatenate([exp.pd_split(s, key) for s in ("train", "validation", "test")])
else:
    frame, y, p = exp.frame_split(scope), exp.y_split(scope), exp.pd_split(scope, key)
port_all = pa.build_portfolio(frame, y, p, exp.exposure_col, fin["lgd"], fin["ccf"], fin["default_exposure"], cut)

# ---------------------------------------------------------------- Filters
cat_candidates = [c for c in frame.columns if c not in (exp.target["column"],) and dv.infer_kind(frame[c]) == dv.KIND_CATEGORICAL
                  and 1 < frame[c].nunique() <= 40 and c not in exp.sensitive_cols]
with st.expander("Filters", expanded=False):
    f1, f2 = st.columns(2)
    filter_cols = f1.multiselect("Filter by", cat_candidates, default=[c for c in ("NAME_CONTRACT_TYPE",) if c in cat_candidates])
    cat_filters = {}
    for c in filter_cols:
        levels = sorted(port_all[c].dropna().astype(str).unique())
        cat_filters[c] = f2.multiselect(c, levels, default=levels, key=f"flt_{c}")
    lo, hi = float(port_all["ead"].min()), float(port_all["ead"].max())
    ead_range = st.slider("Exposure range", lo, hi, (lo, hi)) if hi > lo else None
    date_range = None
    if exp.date_col:
        d = dv.parse_dates(port_all[exp.date_col]).dropna()
        if len(d):
            date_range = st.slider("Application date", d.min().date(), d.max().date(), (d.min().date(), d.max().date()))
port = pa.filter_portfolio(port_all, cat_filters, ead_range, exp.date_col, date_range)
if port.empty:
    ui.empty_state("No loans match these filters", "Widen the filters to see results.")
    st.stop()

k = pa.portfolio_kpis(port)
ui.calibration_warning(exp.models.get(key))
if ss.get("use_loaded_bundle"):
    ui.note(f"A model bundle loaded from disk is active for scoring, but portfolio analysis needs labelled held-out data, so this "
            f"page uses this session's <b>{exp.models[key].label}</b> on its own splits.")
synthetic = exp.dataset.get("is_synthetic")
ui.tiles([
    {"label": "Loans evaluated", "value": ui.num(k["n_loans"]), "prov": "observed", "sub": f"of {len(port_all):,} in scope"},
    {"label": "Total exposure at default", "value": ui.money(k["total_ead"]), "prov": "assumption" if not exp.exposure_col else "observed",
     "help": "Σ exposure × CCF", "sub": exp.exposure_col or "fixed exposure per loan"},
    {"label": "Observed default rate", "value": ui.pct(k["observed_default_rate"], 2), "prov": "observed",
     "help": "Share of these loans whose historical label is default.", "sub": f"EAD-weighted {ui.pct(k['observed_default_rate_ead_weighted'], 2)}"},
    {"label": "Average predicted PD", "value": ui.pct(k["avg_pd"], 2), "prov": "model",
     "help": "Mean calibrated PD from the active model.", "sub": f"EAD-weighted {ui.pct(k['ead_weighted_pd'], 2)}"},
    {"label": "12-month expected credit loss", "value": ui.money(k["total_el"]), "prov": "simulated",
     "help": "Σ PD × LGD × EAD over these loans, treating each model PD as a 12-month probability (an assumption). "
             "Multi-year horizons are handled on the Financial Scenario Lab page.", "sub": f"{ui.pct(k['el_rate'], 2)} of EAD"},
    {"label": "High-risk share", "value": ui.pct(k["high_risk_share_count"]), "prov": "model",
     "help": "Share of loans in the Elevated or High band.", "sub": f"{ui.pct(k['high_risk_share_ead'])} of exposure"},
])
ui.provenance_legend(["observed", "model", "simulated", "assumption"])

bands = pa.band_summary(port)
c1, c2 = st.columns(2)
with c1:
    fig = go.Figure()
    fig.add_bar(x=bands["risk_band"].astype(str), y=bands["ead_share"], name="Share of exposure", marker_color="#8A9BB3")
    fig.add_bar(x=bands["risk_band"].astype(str), y=bands["el_share"], name="Share of expected loss",
                marker_color=[ui.BAND_COLORS.get(str(b), "#1E4C8A") for b in bands["risk_band"]])
    fig.update_layout(title="Exposure vs expected loss by risk band", barmode="group", yaxis_tickformat=".0%")
    ui.plot(fig)
with c2:
    dec = pa.pd_decile_view(port)
    if not dec.empty:
        fig = go.Figure()
        fig.add_bar(x=dec["decile"], y=dec["observed_default_rate"], name="Observed default rate", marker_color="#1E4C8A")
        fig.add_scatter(x=dec["decile"], y=dec["avg_pd"], name="Average predicted PD", mode="lines+markers", line_color="#C2873A")
        fig.update_layout(title="Observed default rate vs predicted PD by PD decile", xaxis=dict(title="PD decile (10 = riskiest)", dtick=1),
                          yaxis_tickformat=".0%")
        ui.plot(fig)
ui.show_table(bands, pct_cols=["observed_default_rate", "avg_pd", "loan_share", "ead_share", "el_share", "el_rate"],
              money_cols=["ead", "expected_loss"], int_cols=["loans"])

# ---------------------------------------------------------------- Segments
st.subheader("Segments and concentration")
seg_candidates = [c for c in frame.columns if c != exp.target["column"] and (
    (dv.infer_kind(frame[c]) == dv.KIND_CATEGORICAL and frame[c].nunique() <= 80) or dv.infer_kind(frame[c]) == dv.KIND_NUMERIC)]
default_seg = next((c for c in ("NAME_INCOME_TYPE", "NAME_CONTRACT_TYPE", "purpose", "grade") if c in seg_candidates), seg_candidates[0])
seg_col = st.selectbox("Segment by", seg_candidates, index=seg_candidates.index(default_seg))
seg = pa.segment_summary(port, seg_col)
conc = pa.concentration_metrics(seg)
c1, c2 = st.columns([1.5, 1])
with c1:
    s = seg.head(15).iloc[::-1]
    fig = go.Figure()
    fig.add_bar(y=s[seg_col].astype(str), x=s["expected_loss"], orientation="h", name="Expected loss", marker_color="#A23B3B",
                customdata=s[["el_rate", "observed_default_rate", "loans"]],
                hovertemplate="%{y}<br>EL %{x:,.0f}<br>EL/EAD %{customdata[0]:.2%}<br>Observed DR %{customdata[1]:.2%}<br>%{customdata[2]:,} loans<extra></extra>")
    fig.update_layout(title=f"Expected loss by {seg_col}", xaxis_title=ui.amount_label("Expected loss"))
    ui.plot(fig, height=max(320, 26 * len(s) + 100))
with c2:
    ui.tiles([
        {"label": "Largest segment share of EAD", "value": ui.pct(conc["top1_share"]), "prov": "observed"},
        {"label": "Top-3 share of EAD", "value": ui.pct(conc["top3_share"]), "prov": "observed"},
        {"label": "Herfindahl index", "value": ui.dec(conc["hhi"], 3), "prov": "observed",
         "help": "Σ (segment share of exposure)². 1 = everything in one segment; lower = more diversified.",
         "sub": f"≈ {ui.dec(conc['effective_segments'], 1)} equal-sized segments"},
    ])
    lc = pa.loss_concentration_curve(port)
    fig = go.Figure()
    fig.add_scatter(x=lc["share_of_loans"], y=lc["share_of_expected_loss"], mode="lines", name="Expected loss", line_color="#A23B3B", fill="tozeroy",
                    fillcolor="rgba(162,59,59,0.08)")
    fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", name="Equal loss per loan", line=dict(color="#B8C4D4", dash="dot"))
    top10 = float(np.interp(0.10, lc["share_of_loans"], lc["share_of_expected_loss"]))
    ui.note(f"The riskiest 10% of loans (by expected loss) hold <b>{ui.pct(top10, 0)}</b> of total expected loss.")
    fig.update_layout(title="Loss concentration curve",
                      xaxis=dict(title="Share of loans (highest EL first)", tickformat=".0%"), yaxis=dict(title="Share of expected loss", tickformat=".0%"))
    ui.plot(fig, height=300)
ui.show_table(seg, pct_cols=["observed_default_rate", "avg_pd", "loan_share", "ead_share", "el_share", "el_rate"],
              money_cols=["ead", "expected_loss"], int_cols=["loans"], height=320)

# ---------------------------------------------------------------- Size, income, distribution, time
c1, c2 = st.columns(2)
with c1:
    size = pa.summarise(port.assign(size_band=pa.quantile_bands(port["ead"], 5)), "size_band")
    size = size[size["size_band"] != "Missing"]
    fig = go.Figure()
    fig.add_bar(x=size["size_band"].astype(str), y=size["el_rate"], name="EL / EAD", marker_color="#5D4A8A")
    fig.add_scatter(x=size["size_band"].astype(str), y=size["observed_default_rate"], name="Observed default rate", mode="lines+markers",
                    line_color="#1E4C8A")
    fig.update_layout(title="By loan-size band (EAD quintiles)", yaxis_tickformat=".1%")
    ui.plot(fig)
with c2:
    inc_col = next((c for c in ("AMT_INCOME_TOTAL", "annual_inc", "income") if c in port.columns), None)
    if inc_col:
        inc = pa.summarise(port.assign(income_band=pa.quantile_bands(port[inc_col], 5)), "income_band")
        inc = inc[inc["income_band"] != "Missing"]
        fig = go.Figure()
        fig.add_bar(x=inc["income_band"].astype(str), y=inc["el_rate"], name="EL / EAD", marker_color="#5D4A8A")
        fig.add_scatter(x=inc["income_band"].astype(str), y=inc["observed_default_rate"], name="Observed default rate",
                        mode="lines+markers", line_color="#1E4C8A")
        fig.update_layout(title=f"By income band ({inc_col} quintiles)", yaxis_tickformat=".1%")
        ui.plot(fig)
    else:
        h = go.Figure()
        h.add_histogram(x=port["pd"], marker_color="#1E4C8A", nbinsx=40)
        h.update_layout(title="Distribution of predicted PD", xaxis_tickformat=".0%", yaxis_title="Loans")
        ui.plot(h)

if exp.date_col:
    tt = pa.time_trend(port, exp.date_col, "M")
    if not tt.empty:
        fig = go.Figure()
        fig.add_bar(x=tt["period"], y=tt["ead"], name="EAD", marker_color="#D5DEEA", yaxis="y2")
        fig.add_scatter(x=tt["period"], y=tt["observed_default_rate"], name="Observed default rate", line_color=ui.DEFAULT_COLOR)
        fig.add_scatter(x=tt["period"], y=tt["avg_pd"], name="Average predicted PD", line_color="#C2873A")
        fig.update_layout(title="Monthly trend", yaxis=dict(tickformat=".0%"), yaxis2=dict(overlaying="y", side="right", showgrid=False))
        ui.plot(fig)

st.subheader("Export")
c1, c2 = st.columns(2)
with c1:
    summary = pd.concat([bands.assign(dimension="risk_band").rename(columns={"risk_band": "segment"}),
                         seg.assign(dimension=seg_col).rename(columns={seg_col: "segment"})], ignore_index=True)
    ui.download("Portfolio summary (CSV)", summary.to_csv(index=False), "portfolio_summary.csv", "text/csv")
with c2:
    id_col = next((c for c, r in exp.excluded.items() if r == "identifier" and c in port.columns), None)
    cols = ([id_col] if id_col else []) + ["observed_default", "pd", "risk_band", "ead", "lgd", "expected_loss"]
    ui.download("Loan-level scores (CSV)", port[cols].to_csv(index=False), "portfolio_loans.csv", "text/csv",
                help="Identifier, observed label, PD, band, EAD, LGD and EL — no other borrower attributes.")
if synthetic:
    ui.note("All values above are computed on synthetic data.")
