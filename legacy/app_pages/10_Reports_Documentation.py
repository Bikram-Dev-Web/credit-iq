"""Page 10 — Reports & Documentation: generated report, model card, exports, docs, data management."""
import pandas as pd
import streamlit as st

from src import config
from src import evaluation as ev
from src import financial_calculations as fc
from src import persistence as ps
from src import portfolio_analysis as pa
from src import reporting as rp
from src import state, ui

ss = st.session_state
ui.page_header("Reports & Documentation", "Export what you have actually computed — every section is labelled as a measured result, "
               "an observation, an assumption or a simulation.")
ui.provenance_legend(["measured", "observed", "assumption", "simulated"])

tab_rep, tab_card, tab_docs, tab_data = ui.tabs(["Project report", "Model card", "Documentation", "Data & privacy"], key="tabs_reports_documentation")


def build_context() -> dict:
    ctx: dict = {}
    if state.has_data():
        ctx["dataset"] = ss.ds_info.to_dict()
        cache = ss.get("_profile_cache", {}).get(ss.ds_info.fingerprint)
        if cache:
            summary, _, issues, _ = cache
            ctx["quality"] = {**{k: v for k, v in summary.items() if k != "memory_mb"},
                              "warnings": sum(i.severity == "warning" for i in issues)}
    if state.has_target():
        ctx["target"] = ss.target.to_dict()
    exp = ss.experiment
    if exp is not None:
        key = ss.active_model_key or exp.selected
        ctx["experiment"] = exp
        tab = ev.calibration_table(exp.y_split("test"), exp.pd_split("test", key), 10)
        ctx["calibration_table"] = tab[["bin", "n", "mean_pd", "observed_rate", "ci_low", "ci_high"]]
        ctx["ece"] = ev.expected_calibration_error(tab)
        if ss.perm_importance is not None and ss.perm_importance[0] == key:
            ctx["explanations"] = ss.perm_importance[1].head(15)[["feature", "auc_drop_mean", "auc_drop_std"]]
        fin = state.fin()
        ctx["assumptions"] = {"LGD": fin["lgd"], "CCF": fin["ccf"], "Annual lending rate": fin["annual_rate"],
                              "Annual funding rate": fin["funding_rate"], "Operating cost per loan per year": fin["opex_per_loan"],
                              "Horizon (years)": fin["horizon_years"], "Exposure column": exp.exposure_col or "fixed per loan",
                              "Risk-band cutoffs (PD)": ", ".join(f"{c:.1%}" for c in ss.band_cutoffs),
                              "Decision threshold (PD)": f"{state.experiment_threshold():.2%}",
                              "PD convention": "model PD treated as a 12-month probability; PD over h years = 1 − (1 − PD)^h"}
        port = pa.build_portfolio(exp.frame_split("test"), exp.y_split("test"), exp.pd_split("test", key), exp.exposure_col,
                                  fin["lgd"], fin["ccf"], fin["default_exposure"], ss.band_cutoffs)
        ctx["portfolio_kpis"] = pa.portfolio_kpis(port)
        ctx["band_table"] = pa.band_summary(port)[["risk_band", "loans", "ead", "observed_default_rate", "avg_pd", "expected_loss", "el_rate"]]
        if ss.get("contribution_result"):
            ctx["contribution"] = {k: v for k, v in ss.contribution_result.items() if k != "assumptions"}
        ctx["scenarios"] = (ss.scenario_table if ss.scenario_table is not None else
                            fc.stress_test(port["pd"], fin["lgd"], port["ead"], fc.DEFAULT_SCENARIOS, fin["annual_rate"],
                                           fin["funding_rate"], fin["opex_per_loan"], fin["horizon_years"]))
        ctx["scenarios"] = ctx["scenarios"][["scenario", "pd_multiplier", "lgd_shift", "ead_multiplier", "avg_pd", "expected_loss",
                                             "el_rate", "expected_contribution"]]
        if ss.policy_table is not None:
            ctx["policies"] = ss.policy_table[["policy", "tp", "fp", "tn", "fn", "approval_rate", "total_cost",
                                               "realized_contribution_approved_book"]]
        if ss.fairness_table is not None:
            ctx["fairness"] = ss.fairness_table[["group", "n", "observed_default_rate", "mean_pd", "within_threshold_rate", "tpr_recall",
                                                 "fpr_good_flagged", "roc_auc"]]
            ctx["fairness_attribute"] = ss.fairness_attr
    return ctx


with tab_rep:
    if not state.has_data():
        ui.empty_state("Nothing to report yet", "Load data and train a model; the report is built only from results that exist.",
                       "app_pages/2_Data_Center.py", "Open the Data Center")
    else:
        ctx = build_context()
        sections = rp.build_report(ctx)
        md = rp.to_markdown(sections)
        html_doc = rp.to_html(sections)
        missing = []
        if ss.experiment is None:
            missing.append("model results (train in the AI Model Lab)")
        if ss.policy_table is None:
            missing.append("policy comparison (Financial Scenario Lab)")
        if ss.fairness_table is None:
            missing.append("fairness audit (Explainability & Responsible AI)")
        if ss.perm_importance is None:
            missing.append("permutation importance (Explainability)")
        if missing:
            ui.note("Not included yet because not computed: " + "; ".join(missing) + ".")
        stamp = ss.experiment.experiment_id if ss.experiment is not None else "data"
        c1, c2, c3 = st.columns(3)
        with c1:
            ui.download("Report (HTML)", html_doc, f"creditiq_report_{stamp}.html", "text/html")
        with c2:
            ui.download("Report (Markdown)", md, f"creditiq_report_{stamp}.md", "text/markdown")
        with c3:
            if st.button("Save both to reports/", icon=":material/save:"):
                p1 = ps.save_report(md, f"creditiq_report_{stamp}.md")
                p2 = ps.save_report(html_doc, f"creditiq_report_{stamp}.html")
                st.success(f"Saved {p1.name} and {p2.name} in {p1.parent}")
        if ss.experiment is not None:
            exp = ss.experiment
            c1, c2, c3 = st.columns(3)
            with c1:
                ui.download("All metrics (JSON)", ps.to_json(exp.metrics, indent=2), f"metrics_{exp.experiment_id}.json", "application/json")
            with c2:
                ui.download("Test predictions (CSV)", exp.test_predictions().to_csv(index=False), f"test_predictions_{exp.experiment_id}.csv", "text/csv")
            with c3:
                ui.download("Experiment metadata (JSON)", ps.to_json(exp.metadata(), indent=2), f"experiment_{exp.experiment_id}.json", "application/json")
        st.divider()
        for sec in sections:
            with st.expander(f"{sec.title} — {rp.KINDS[sec.kind]}"):
                for p in sec.paragraphs:
                    st.markdown(p)
                for b in sec.bullets:
                    st.markdown(f"- {b}")
                for caption, df in sec.tables:
                    st.markdown(f"**{caption}**")
                    st.dataframe(df, hide_index=True)

with tab_card:
    bundle = state.active_bundle()
    if bundle is None:
        ui.empty_state("No model yet", "The model card is generated from a trained or loaded model.", "app_pages/4_Model_Lab.py", "Open the AI Model Lab")
    else:
        comp = ss.experiment.comparison_table() if ss.experiment is not None and not ss.use_loaded_bundle else None
        card = rp.model_card_markdown(bundle.metadata, comp)
        ui.download("Model card (Markdown)", card, f"model_card_{bundle.metadata.get('experiment_id')}.md", "text/markdown")
        st.markdown(card)

with tab_docs:
    docs = {"README": config.PROJECT_ROOT / "README.md"}
    docs.update({p.stem.replace("_", " ").title(): p for p in sorted(config.DOCS_DIR.glob("*.md"))})
    name = st.selectbox("Document", list(docs))
    path = docs[name]
    if path.exists():
        text = path.read_text(encoding="utf-8")
        ui.download(f"Download {path.name}", text, path.name, "text/markdown")
        st.markdown(text)
    else:
        st.error(f"{path} not found.")

with tab_data:
    st.markdown("**Where things are stored on this computer** (plain files; not encrypted)")
    ui.show_table(pd.DataFrame([{"what": k, "location": v} for k, v in ps.storage_locations().items()]))
    ui.note("Uploaded CSVs are held in memory for the session only and are never written to disk by the app. The audit table "
            "stores scores and a one-way hash of the inputs, not the inputs themselves. Nothing is sent to external services; "
            "Streamlit usage statistics are disabled in .streamlit/config.toml.")
    st.markdown("**Clear local records**")
    c1, c2, c3, c4, c5 = st.columns(5)
    opts = {"experiments": c1.checkbox("Experiment log"), "assessments": c2.checkbox("Assessment audit log"),
            "monitoring": c3.checkbox("Monitoring history"), "models": c4.checkbox("Saved model files"),
            "reports": c5.checkbox("Saved reports")}
    sure = st.checkbox("I understand this permanently deletes the selected local records.")
    if st.button("Delete selected", type="primary", disabled=not (sure and any(opts.values())), icon=":material/delete_forever:"):
        removed = ps.clear_local_records(**opts)
        st.success("Deleted: " + ", ".join(f"{v} {k.replace('_', ' ')}" for k, v in removed.items()))
    if st.button("Clear everything in this browser session (data, target, models)"):
        for k in list(ss.keys()):
            del ss[k]
        st.rerun()
    assess = ps.list_assessments(50)
    if not assess.empty:
        with st.expander(f"Recent assessment audit records ({len(assess)})"):
            ui.show_table(assess, pct_cols=["pd", "threshold"], dec_cols=["raw_score"])

