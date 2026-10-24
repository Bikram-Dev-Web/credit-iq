"""Page 1 — Executive Overview."""
import streamlit as st

from src import data_loader as dl
from src import evaluation as ev
from src import state, ui
from src.portfolio_analysis import build_portfolio, portfolio_kpis

ss = st.session_state
ui.page_header("CreditIQ — AI Credit Risk & Lending Intelligence",
               "From borrower risk prediction to explainable, financially informed lending decisions.")

exp = ss.experiment
bundle = state.active_bundle()
done = [state.has_data(), state.has_target(), state.has_features(), bundle is not None,
        ss.policy_table is not None or ss.scenario_table is not None]
subs = [
    f"{ss.ds_info.n_rows_loaded:,} rows loaded" if state.has_data() else "Synthetic sample or your CSV",
    f"{ss.target.column}: {ui.pct(ss.target.prevalence)} default" if state.has_target() else "Confirm what default means",
    f"{len(ss.predictors)} predictors" if state.has_features() else "Exclude IDs, leakage, sensitive",
    bundle.model.label if bundle is not None else "Train, calibrate, validate",
    "Policies or stress tested" if done[4] else "Expected loss, policies, stress",
]
titles = ["Load data", "Map the target", "Choose features", "Train and validate", "Decide and stress-test"]
next_i = next((i for i, d in enumerate(done) if not d), None)
ui.steps([(t, s, "done" if d else ("next" if i == next_i else "todo")) for i, (t, s, d) in enumerate(zip(titles, subs, done))])

if not state.has_data():
    ui.callout("Start with the bundled <b>synthetic</b> Home Credit–style sample (20,000 fabricated applications) to see "
               "the full workflow in a few minutes. You can switch to the real Kaggle file or your own CSV at any time.")
    c1, c2 = st.columns([1, 2])
    with c1:
        if st.button("Load the synthetic demo dataset", type="primary", icon=":material/play_arrow:"):
            df, info = dl.load_synthetic_dataset()
            state.set_dataset(df, info)
            st.switch_page("app_pages/2_Data_Center.py")
    with c2:
        st.page_link("app_pages/2_Data_Center.py", label="Upload a CSV or open a local file instead", icon=":material/upload_file:")
else:
    info = ss.ds_info
    items = [{"label": "Applications in dataset", "value": ui.num(info.n_rows_loaded), "prov": "observed",
              "sub": info.name if len(info.name) < 34 else info.name[:31] + "…"}]
    if state.has_target():
        t = ss.target
        items.append({"label": "Observed default rate", "value": ui.pct(t.prevalence), "prov": "observed",
                      "help": "Share of labelled applications whose historical outcome is default (per your confirmed mapping).",
                      "sub": f"{t.n_positive:,} defaults"})
    if exp is not None:
        key = state.experiment_model_key()
        m = exp.metrics[key]["test"]
        items.append({"label": "Test ROC-AUC (active model)", "value": ui.dec(m["roc_auc"], 3),
                      "prov": "synthetic" if info.is_synthetic else "measured",
                      "help": ev.METRIC_HELP["roc_auc"], "sub": exp.models[key].label})
        items.append({"label": "Average predicted PD (test)", "value": ui.pct(m["mean_pd"]), "prov": "model",
                      "help": "Mean calibrated probability of default on held-out applications.",
                      "sub": f"vs {ui.pct(m['prevalence'])} observed"})
        fin = state.fin()
        port = build_portfolio(exp.frame_split("test"), exp.y_split("test"), exp.pd_split("test", key),
                               exp.exposure_col, fin["lgd"], fin["ccf"], fin["default_exposure"], ss.band_cutoffs)
        k = portfolio_kpis(port)
        items.append({"label": "12-month expected credit loss (test book)", "value": ui.money(k["total_el"]), "prov": "simulated",
                      "help": "Σ PD × LGD × EAD over held-out loans, using the LGD and exposure assumptions and treating "
                              "the model PD as a 12-month probability.",
                      "sub": f"{ui.pct(k['el_rate'], 2)} of {ui.money(k['total_ead'])} EAD"})
    ui.tiles(items)
    if exp is not None:
        ui.calibration_warning(exp.models[key])

    if exp is not None:
        st.subheader("Current experiment")
        comp = exp.comparison_table()
        view = comp[["model", "selected", "val_roc_auc", "test_roc_auc", "test_pr_auc", "test_brier", "threshold", "test_recall", "test_precision"]]
        ui.show_table(view, pct_cols=["threshold", "test_recall", "test_precision"],
                      dec_cols=["val_roc_auc", "test_roc_auc", "test_pr_auc", "test_brier"])
        cfg = exp.config
        ui.note(f"Experiment {exp.experiment_id} · {exp.created_at}. Split: {exp.split.strategy} "
                f"({exp.split.sizes()['train']:,} / {exp.split.sizes()['validation']:,} / {exp.split.sizes()['test']:,}). "
                f"Calibration: {cfg.calibration}. Threshold rule: {ev.THRESHOLD_RULES[cfg.threshold_rule]}.")
    elif bundle is not None:
        ui.callout(f"A saved model is active: <b>{bundle.model.label}</b> from experiment {bundle.metadata.get('experiment_id')}. "
                   "Scoring and monitoring work; pages that need held-out rows ask you to train in this session.")

st.subheader("Where to go next")
cols = st.columns(3)
links = [
    ("app_pages/2_Data_Center.py", "Data Center", ":material/database:", "Profile data, map the target, screen for leakage."),
    ("app_pages/3_EDA.py", "Exploratory Analysis", ":material/query_stats:", "Default rates by feature, segment and over time."),
    ("app_pages/4_Model_Lab.py", "AI Model Lab", ":material/model_training:", "Train three algorithms; compare on held-out data."),
    ("app_pages/5_Borrower_Simulator.py", "Borrower Risk Simulator", ":material/person_search:", "Score a hypothetical applicant and see why."),
    ("app_pages/6_Portfolio_Intelligence.py", "Portfolio Intelligence", ":material/pie_chart:", "Exposure, expected loss and concentration."),
    ("app_pages/7_Financial_Scenarios.py", "Financial Scenario Lab", ":material/calculate:", "Contribution, stress tests, policies, NPV."),
    ("app_pages/8_Explainability_Fairness.py", "Explainability & Responsible AI", ":material/balance:", "Feature attribution and group-level checks."),
    ("app_pages/9_Model_Monitoring.py", "Model Monitoring", ":material/monitor_heart:", "Drift and data-quality checks on new data."),
    ("app_pages/10_Reports_Documentation.py", "Reports & Documentation", ":material/description:", "Export the report, model card and results."),
]
for i, (path, label, icon, desc) in enumerate(links):
    with cols[i % 3]:
        st.page_link(path, label=label, icon=icon)
        st.caption(desc)

with st.expander("Demonstration script (about 10 minutes)"):
    st.markdown(
        """
1. **Executive Overview** → *Load the synthetic demo dataset*.
2. **Data Center → Target mapping**: TARGET, default label `1`, tick the confirmation, *Confirm target mapping*.
3. **Features & leakage**: run the leakage screen — `POST_LOAN_COLLECTION_CALLS` is flagged and excluded; save the features.
4. **AI Model Lab**: train all three models (≈ 10–30 s on the sample).
5. Compare ROC-AUC, PR-AUC and Brier score on the **test** split; open the calibration curve.
6. **Threshold explorer**: move the threshold and watch the confusion matrix, recall and precision change.
7. **Borrower Risk Simulator**: score a typical applicant, then raise the loan amount or lower `EXT_SOURCE_2`.
8. Read the explanation: which features pushed the score up or down.
9. **Portfolio Intelligence**: expected credit loss by risk band and segment.
10. **Financial Scenario Lab**: lending economics, break-even PD and an adverse stress scenario.
11. **Policy comparison**: approve-all vs fixed threshold vs cost-minimising vs expected-contribution rule.
12. **Reports & Documentation**: download the HTML report and model card.
"""
    )
