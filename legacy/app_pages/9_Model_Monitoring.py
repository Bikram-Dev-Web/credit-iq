"""Page 9 — Model Monitoring: compare new data with the training reference."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import config
from src import data_loader as dl
from src import data_validation as dv
from src import evaluation as ev
from src import monitoring as mon
from src import persistence as ps
from src import state, ui

ss = st.session_state
ui.page_header("Model Monitoring", "Check whether new applications still look like the data the model was trained on, and — when "
               "outcomes are known — whether it still performs.")
ui.require_model()

bundle = state.active_bundle()
model = bundle.model
exp_id = bundle.metadata.get("experiment_id")
target = bundle.metadata.get("target", {})

st.subheader("Run a check")
sources = ["Bundled synthetic later-period batch", "Upload a CSV batch"]
if ss.experiment is not None and not ss.use_loaded_bundle:
    sources.append("Held-out test set (baseline: expect little drift)")
c1, c2 = st.columns([2, 1])
src = c1.radio("New data", sources)
upload = c1.file_uploader("CSV with the model's input columns (target optional)", type=["csv"]) if src.startswith("Upload") else None
if src.startswith("Bundled"):
    c2.markdown("<div class='ciq-note'>Synthetic applications dated January–June 2026, generated with a deliberate shift "
                "(weaker credit profile, lower incomes, more missing external scores). Labelled synthetic.</div>", unsafe_allow_html=True)

if st.button("Run monitoring check", type="primary", icon=":material/monitor_heart:"):
    try:
        if src.startswith("Bundled"):
            batch, name = dl.load_synthetic_monitoring_batch(), "SYNTHETIC_monitoring_batch_2026H1"
            synthetic = True
        elif src.startswith("Upload"):
            if upload is None:
                st.error("Choose a CSV file first.")
                st.stop()
            if upload.size > config.MAX_UPLOAD_MB * 1e6:
                st.error(f"File exceeds {config.MAX_UPLOAD_MB} MB.")
                st.stop()
            batch, info = dl.read_csv_file(upload.getvalue(), name=upload.name)
            name, synthetic = info.name, info.is_synthetic
        else:
            batch, name = ss.experiment.frame_split("test"), "held_out_test_set"
            synthetic = bool(ss.experiment.dataset.get("is_synthetic"))
        schema = mon.schema_check(model.input_columns, batch, bundle.reference)
        if schema["n_present"] == 0:
            st.error("None of the model's input columns are in this file — it is not compatible with the active model.")
            st.stop()
        drift = mon.feature_drift(bundle.reference, batch)
        pd_new = model.predict_pd(batch)
        score = mon.score_drift(bundle.reference, pd_new)
        perf = None
        tcol = target.get("column")
        if tcol and tcol in batch.columns:
            y_new = dv.apply_target_mapping(batch[tcol], target.get("positive_values (default)", []), target.get("negative_values (non-default)", []))
            mask = y_new.notna().to_numpy()
            if mask.sum() >= 20 and y_new[mask].nunique() == 2:
                perf = ev.classification_metrics(y_new[mask].astype(int), pd_new[mask], state.active_threshold())
                perf["reference_roc_auc"] = bundle.metadata.get("metrics", {}).get(model.key, {}).get("test", {}).get("roc_auc")
        alerts = mon.drift_alerts(drift, score, schema)
        if perf and perf["reference_roc_auc"] and perf["roc_auc"] is not None and perf["roc_auc"] < perf["reference_roc_auc"] - 0.03:
            alerts.append(f"ROC-AUC fell from {perf['reference_roc_auc']:.3f} (test) to {perf['roc_auc']:.3f} on this batch.")
        if perf and abs(perf["mean_pd"] - perf["prevalence"]) > 0.25 * max(perf["prevalence"], 0.01):
            alerts.append(f"Calibration-in-the-large: mean PD {perf['mean_pd']:.2%} vs observed default rate {perf['prevalence']:.2%}.")
        ss.monitor_result = {"name": name, "n": len(batch), "schema": schema, "drift": drift, "score": score, "perf": perf,
                             "alerts": alerts, "exp_id": exp_id, "model": model.key,
                             "synthetic": synthetic}
        ps.log_monitoring_run({"experiment_id": exp_id, "model": model.key, "batch_name": name, "n_rows": len(batch),
                               "max_feature_psi": float(drift["psi"].max()) if drift["psi"].notna().any() else None,
                               "n_significant": int((drift["status"] == "Significant shift").sum()),
                               "score_psi": score["psi"] if score else None, "mean_pd": float(np.mean(pd_new)),
                               "observed_default_rate": perf["prevalence"] if perf else None,
                               "roc_auc": perf["roc_auc"] if perf else None, "alerts": alerts})
    except (ValueError, KeyError, pd.errors.ParserError) as exc:
        st.error(f"Monitoring check failed: {exc}")

r = ss.monitor_result
if r and r["exp_id"] == exp_id and r["model"] == model.key:
    st.divider()
    st.subheader(f"Results for {r['name']}")
    if r["alerts"]:
        ui.callout("<b>Warnings</b><br>" + "<br>".join(r["alerts"]), "risk")
    else:
        ui.callout("No material shift detected against the training reference.", "ok")
    d, s, p = r["drift"], r["score"], r["perf"]
    prov = "synthetic" if r.get("synthetic") or r["name"].startswith("SYNTHETIC") else "observed"
    items = [
        {"label": "Rows checked", "value": ui.num(r["n"]), "prov": prov},
        {"label": "Inputs present", "value": f"{r['schema']['n_present']} / {r['schema']['n_input_columns']}", "prov": prov},
        {"label": "Features with significant shift", "value": ui.num((d["status"] == "Significant shift").sum()), "prov": prov,
         "help": f"Population Stability Index ≥ {config.PSI_SIGNIFICANT}. Rule of thumb: < {config.PSI_MODERATE} stable, "
                 f"{config.PSI_MODERATE}–{config.PSI_SIGNIFICANT} moderate, ≥ {config.PSI_SIGNIFICANT} significant."},
        {"label": "PD distribution PSI", "value": ui.dec(s["psi"], 3) if s else "n/a", "prov": "model", "sub": s["status"] if s else None},
        {"label": "Mean PD: new vs reference", "value": f"{ui.pct(s['new_mean_pd'], 1)} vs {ui.pct(s['ref_mean_pd'], 1)}" if s else "n/a", "prov": "model"},
    ]
    if p:
        items += [{"label": "Observed default rate (batch)", "value": ui.pct(p["prevalence"], 2), "prov": prov},
                  {"label": "ROC-AUC on batch", "value": ui.dec(p["roc_auc"], 3), "prov": prov, "sub": f"test was {ui.dec(p['reference_roc_auc'], 3)}"}]
    ui.tiles(items)
    if not p:
        ui.note("No usable outcome labels in this batch, so only input and score drift are assessed. In practice, outcomes "
                "arrive months after origination.")
    c1, c2 = st.columns(2)
    with c1:
        top = d.dropna(subset=["psi"]).head(20).iloc[::-1]
        colors = ["#A23B3B" if v >= config.PSI_SIGNIFICANT else "#C2873A" if v >= config.PSI_MODERATE else "#2F6B5E" for v in top["psi"]]
        fig = go.Figure(go.Bar(x=top["psi"], y=top["column"], orientation="h", marker_color=colors))
        fig.add_vline(x=config.PSI_MODERATE, line_dash="dot", line_color="#C2873A")
        fig.add_vline(x=config.PSI_SIGNIFICANT, line_dash="dot", line_color="#A23B3B")
        fig.update_layout(title="Population Stability Index by input (top 20)", xaxis_title="PSI")
        ui.plot(fig, height=max(340, 22 * len(top) + 90))
    with c2:
        if s:
            cuts = [0.0] + list(s["cuts"]) + [1.0]
            labels = [f"{cuts[i]:.1%}–{cuts[i + 1]:.1%}" for i in range(len(cuts) - 1)]
            fig = go.Figure()
            fig.add_bar(x=labels, y=s["ref_props"][: len(labels)], name="Reference (validation)", marker_color="#8A9BB3")
            fig.add_bar(x=labels, y=s["new_props"][: len(labels)], name="New batch", marker_color="#1E4C8A")
            fig.update_layout(title="Predicted PD distribution (reference deciles)", barmode="group", yaxis_tickformat=".0%",
                              xaxis_title="PD bin")
            ui.plot(fig, height=360)
    ui.show_table(d, dec_cols=["psi"], pct_cols=["ref_missing_rate", "new_missing_rate", "missing_change"], height=320)
    if r["schema"]["type_changes"]:
        st.warning("Type changes: " + ", ".join(f"{t['column']} ({t['expected']} → {t['found']})" for t in r["schema"]["type_changes"]))
    ui.download("Drift table (CSV)", d.to_csv(index=False), "drift_report.csv", "text/csv")

st.subheader("Monitoring history")
hist = ps.list_monitoring_runs(exp_id)
if hist.empty:
    ui.empty_state("Monitoring history is not available yet", "Each check you run for this model is recorded locally. "
                   "Nothing is shown until real checks exist — no trends are simulated.")
else:
    hist = hist[hist["model"] == model.key]
    fig = go.Figure()
    fig.add_scatter(x=hist["created_at"], y=hist["score_psi"], mode="lines+markers", name="PD distribution PSI", line_color="#1E4C8A")
    fig.add_scatter(x=hist["created_at"], y=hist["max_feature_psi"], mode="lines+markers", name="Largest feature PSI", line_color="#C2873A")
    if hist["roc_auc"].notna().any():
        fig.add_scatter(x=hist["created_at"], y=hist["roc_auc"], mode="lines+markers", name="ROC-AUC (right axis)", yaxis="y2",
                        line=dict(color="#2F6B5E", dash="dot"))
    fig.update_layout(title="Recorded monitoring checks", yaxis_title="PSI", yaxis2=dict(overlaying="y", side="right", showgrid=False, range=[0.5, 1]))
    ui.plot(fig, height=340)
    ui.show_table(hist.drop(columns=["experiment_id"]), dec_cols=["max_feature_psi", "score_psi", "roc_auc"],
                  pct_cols=["mean_pd", "observed_default_rate"])
