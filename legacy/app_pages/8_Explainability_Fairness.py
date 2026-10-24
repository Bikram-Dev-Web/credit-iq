"""Page 8 — Explainability & Responsible AI."""
import numpy as np
import plotly.graph_objects as go
import streamlit as st

from src import explainability as xai
from src import fairness as fair
from src import state, ui

ss = st.session_state
ui.page_header("Explainability & Responsible AI", "Which inputs drive the model, how it treats different groups, and the "
               "safeguards any real use would need.")
ui.require_model()

bundle = state.active_bundle()
model, schema = bundle.model, bundle.schema
exp = ss.experiment if not ss.use_loaded_bundle else None
if exp is not None and (ss.active_model_key or exp.selected) != model.key:
    exp = None

ui.callout("Explanations show <b>what the model has learned to associate</b> with default in this training data. They do not "
           "show that an input <i>causes</i> default, and changing an input in the real world would not necessarily change risk.")

tab_g, tab_l, tab_pd, tab_f, tab_r = ui.tabs(["Global importance", "Local explanation", "Feature effects", "Fairness audit", "Responsible use"], key="tabs_explainability_fairness")

with tab_g:
    imp, method = xai.native_importance(model, bundle.transformed_mean)
    c1, c2 = st.columns(2)
    with c1:
        top = imp.head(20).iloc[::-1]
        fig = go.Figure(go.Bar(x=top["importance"], y=top["feature"], orientation="h", marker_color="#4E7FA8"))
        fig.update_layout(title="Model-internal importance (share of total)", xaxis_tickformat=".0%")
        ui.plot(fig, height=max(340, 22 * len(top) + 90))
        ui.note(f"{method}. Derived ratio features are listed under their own names.")
    with c2:
        if exp is None:
            ui.empty_state("Permutation importance needs validation rows", "Train a model in this session to compute it.")
        else:
            if st.button("Compute permutation importance on validation data", icon=":material/shuffle:"):
                with st.spinner("Shuffling each input and re-scoring (up to 3,000 validation rows, 3 repeats)"):
                    ss.perm_importance = (model.key, xai.permutation_importance_table(model, exp.X_split("validation"), exp.y_split("validation")))
            if ss.perm_importance is not None and ss.perm_importance[0] == model.key:
                pi = ss.perm_importance[1]
                t = pi.head(20).iloc[::-1]
                fig = go.Figure(go.Bar(x=t["auc_drop_mean"], y=t["feature"], orientation="h", marker_color="#1E4C8A",
                                       error_x=dict(type="data", array=t["auc_drop_std"], thickness=1)))
                fig.update_layout(title="Permutation importance (drop in validation ROC-AUC)")
                ui.plot(fig, height=max(340, 22 * len(t) + 90))
                ui.note("Model-agnostic: how much ranking quality falls when an input's values are shuffled. Correlated inputs can "
                        "substitute for each other, so each may look less important than the group.")
    if exp is not None:
        gc_key = (exp.experiment_id, model.key)
        if ss.get("global_contrib") is None or ss.global_contrib[0] != gc_key:
            with st.spinner("Computing average contributions"):
                ss.global_contrib = (gc_key, xai.global_contribution_importance(model, exp.X_split("validation"), schema,
                                                                                 bundle.transformed_mean, max_rows=300))
        gi, meta = ss.global_contrib[1]
        t = gi.head(20).iloc[::-1]
        fig = go.Figure(go.Bar(x=t["mean_abs_contribution"], y=t["feature"], orientation="h", marker_color="#5D4A8A"))
        fig.update_layout(title=f"Mean |contribution| over {meta['rows_used']} validation applicants", xaxis_title=meta["units"])
        ui.plot(fig, height=max(340, 22 * len(t) + 90))
        ui.note(f"Method: {meta['method']}. {meta.get('fallback_reason', '')}")

with tab_l:
    if exp is None:
        ui.empty_state("Pick an applicant in the simulator", "With a model loaded from disk there are no stored applicants; score one in the "
                       "Borrower Risk Simulator to see its explanation.", "app_pages/5_Borrower_Simulator.py", "Open the simulator")
    else:
        Xt, yt, pt = exp.X_split("test"), exp.y_split("test"), exp.pd_split("test", model.key)
        c1, c2 = st.columns([1, 2])
        how = c1.radio("Applicant", ["Highest predicted risk", "Lowest predicted risk", "Choose a test row"])
        if how == "Choose a test row":
            i = int(c1.number_input("Test row", 0, len(Xt) - 1, 0))
        else:
            i = int(np.argmax(pt) if how.startswith("Highest") else np.argmin(pt))
        x = Xt.iloc[[i]]
        expl, meta = xai.local_explanation(model, x, schema, bundle.transformed_mean, top=15)
        with c2:
            ui.tiles([
                {"label": "Calibrated PD" if model.is_calibrated else "Uncalibrated score", "value": ui.pct(pt[i], 2), "prov": "model"},
                {"label": "Raw model score", "value": ui.dec(exp.score_split("test", model.key)[i], 4), "prov": "model"},
                {"label": "Observed outcome", "value": "Defaulted" if yt[i] == 1 else "Did not default", "prov": "observed",
                 "help": "Historical label of this applicant — what actually happened, not a prediction."},
            ])
        e = expl.iloc[::-1]
        fig = go.Figure(go.Bar(x=e["contribution"], y=[f"{f} = {ui.val(v)}" for f, v in zip(e["feature"], e["value"])],
                               orientation="h", marker_color=[ui.DEFAULT_COLOR if v > 0 else "#1E4C8A" for v in e["contribution"]]))
        fig.add_vline(x=0, line_color="#9AA8BC")
        fig.update_layout(title="Contributions for this applicant (red raises the score)", xaxis_title=meta["units"])
        ui.plot(fig, height=max(340, 26 * len(e) + 90))
        base = f" Base value: {ui.dec(meta['base_value'], 4)}." if meta.get("base_value") is not None else ""
        ui.note(f"Method: {meta['method']}.{base} One-hot and missing-indicator columns are summed into their source column. "
                f"{meta.get('fallback_reason', '')}")

with tab_pd:
    if exp is None:
        ui.empty_state("Needs validation rows", "Train a model in this session to compute feature-effect curves.")
    else:
        feats = [f for f in xai.native_importance(model, bundle.transformed_mean)[0]["feature"] if f in model.input_columns] or model.input_columns
        f = st.selectbox("Feature", feats)
        curve = xai.partial_dependence_curve(model, exp.X_split("validation"), f, schema)
        if curve.empty:
            ui.note("Not enough values to draw this curve.")
        else:
            fig = go.Figure()
            x = curve["value"].astype(str) if schema[f]["kind"] != "numeric" else curve["value"]
            fig.add_scatter(x=x, y=curve["p75"], mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip")
            fig.add_scatter(x=x, y=curve["p25"], mode="lines", line=dict(width=0), fill="tonexty", fillcolor="rgba(30,76,138,0.12)",
                            name="Middle 50% of applicants")
            fig.add_scatter(x=x, y=curve["avg_pd"], mode="lines+markers", line_color="#1E4C8A", name="Average PD")
            fig.update_layout(title=f"Average predicted PD as {f} varies (others held at observed values)", yaxis_tickformat=".1%", xaxis_title=f)
            ui.plot(fig, height=400)
            ui.note("Partial dependence over 300 validation applicants. It shows the model's response, including through derived "
                    "ratios built from this input. Values outside the plotted range were rare in training.")

with tab_f:
    if exp is None:
        ui.empty_state("Needs held-out rows", "The audit uses held-out applicants from a model trained in this session.")
    else:
        frame_t = exp.frame_split("test")
        options = [c for c in exp.sensitive_cols if c in frame_t.columns] + \
                  [c for c in frame_t.columns if c not in exp.sensitive_cols and c != exp.target["column"] and frame_t[c].nunique() <= 12]
        if not options:
            ui.empty_state("No audit attribute", "Select attributes for the fairness audit in the Data Center.")
        else:
            c1, c2 = st.columns([1, 2])
            attr = c1.selectbox("Audit attribute", options, help="Sensitive attributes are excluded from the model by default and used only here.")
            in_model = attr in model.input_columns
            c2.markdown(f"<div class='ciq-note' style='margin-top:1.8rem'>{attr} is <b>{'a model input' if in_model else 'not a model input'}</b>. "
                        "Group membership can still be reflected through correlated inputs (proxies).</div>", unsafe_allow_html=True)
            groups = fair.make_groups(frame_t[attr], attr)
            thr = state.active_threshold()
            gm = fair.group_metrics(exp.y_split("test"), exp.pd_split("test", model.key), thr, groups)
            ss.fairness_table, ss.fairness_attr = gm, attr
            small = gm[gm["small_sample"]]
            if len(small):
                st.warning(f"Small groups ({', '.join(small['group'])}) have fewer than {fair.MIN_GROUP_SIZE} applicants or 10 defaults; "
                           "their rates are too noisy to compare and are excluded from the disparity summary.")
            ui.show_table(gm, pct_cols=["share", "observed_default_rate", "mean_pd", "calibration_gap", "flag_rate", "within_threshold_rate",
                                        "tpr_recall", "fnr_missed_defaults", "fpr_good_flagged", "precision"], dec_cols=["roc_auc"], int_cols=["n"])
            g2 = gm[~gm["small_sample"]]
            c1, c2 = st.columns(2)
            with c1:
                fig = go.Figure()
                fig.add_bar(x=g2["group"], y=g2["within_threshold_rate"], name="Below threshold (not flagged)", marker_color="#2F6B5E")
                fig.add_bar(x=g2["group"], y=g2["fpr_good_flagged"], name="Good applicants flagged (FPR)", marker_color="#C2873A")
                fig.add_bar(x=g2["group"], y=g2["tpr_recall"], name="Defaulters caught (TPR)", marker_color="#1E4C8A")
                fig.update_layout(title=f"Decision rates by {attr} at {ui.pct(thr, 1)} threshold", barmode="group", yaxis_tickformat=".0%")
                ui.plot(fig)
            with c2:
                fig = go.Figure()
                fig.add_bar(x=g2["group"], y=g2["observed_default_rate"], name="Observed default rate", marker_color="#8A9BB3")
                fig.add_bar(x=g2["group"], y=g2["mean_pd"], name="Mean predicted PD", marker_color="#5D4A8A")
                fig.update_layout(title="Calibration by group", barmode="group", yaxis_tickformat=".1%")
                ui.plot(fig)
            ds = fair.disparity_summary(gm)
            if not ds.empty:
                st.markdown("**Largest gaps between groups**")
                ui.show_table(ds, dec_cols=["highest", "lowest", "difference", "ratio_low_to_high"])
                ui.note("Different fairness criteria can conflict when default rates differ between groups (equal approval rates, equal "
                        "error rates and equal calibration cannot all hold at once in general). Which gaps matter is a policy and legal "
                        "judgement; no single number certifies a model as fair.")
            st.markdown("**Possible proxy variables**")
            if st.button("Scan model inputs for association with this attribute"):
                with st.spinner("Measuring associations"):
                    ss.proxy_table = (attr, fair.proxy_scan(frame_t, groups, model.input_columns))
            if ss.get("proxy_table") is not None and ss.proxy_table[0] == attr:
                ui.show_table(ss.proxy_table[1].head(20), dec_cols=["association"])
                ui.note("Strongly associated inputs can let a model reproduce group differences even when the attribute itself is excluded. "
                        "Association alone does not make an input illegitimate; it marks it for review.")
            mg = fair.missingness_by_group(frame_t, groups, model.input_columns)
            if not mg.empty and mg["max_gap"].iloc[0] > 0.05:
                st.markdown("**Missingness differs by group**")
                ui.show_table(mg, pct_cols=[c for c in mg.columns if c != "feature"])

with tab_r:
    st.markdown("""
#### Intended use
An educational decision-support prototype for exploring credit-risk modelling and lending economics. It is **not** a
production credit-decisioning, pricing or regulatory-capital system and must not be used to approve or decline real applicants.

#### Known limitations
- The PD is the probability of the **dataset's own target event** (for Home Credit, payment difficulty on early instalments), not a
  universal or legal definition of default.
- Training data describe one lender's past applicants; performance can degrade for new products, regions or economic conditions.
- Random splits estimate performance on similar applicants, not on future cohorts — prefer a chronological split when dates exist.
- LGD, EAD and costs are user assumptions; expected-loss and contribution figures inherit their uncertainty.
- Explanations are associational and can be unstable for correlated features.

#### Bias and proxy risk
Protected or sensitive attributes (e.g. gender, age, marital status) are excluded from the model by default and used only for the
group-level audit. Other inputs — occupation, organisation type, region, income, housing — can act as proxies. Missing values can be
concentrated in particular groups. Historical labels can encode past lending decisions, not only borrower behaviour.

#### Human-review safeguards for any real deployment
- Treat the model as one input to a credit officer's decision; never as an automatic approval or decline.
- Give every declined or flagged applicant a reason based on legitimate, verifiable factors, and a route to appeal.
- Independent model validation before use: conceptual soundness, out-of-time testing, benchmark comparison.
- Legal review of every input against applicable anti-discrimination and data-protection law in the jurisdiction.
- Ongoing monitoring of drift, calibration and group-level outcomes, with defined thresholds that trigger review or retraining.
- Document data lineage, exclusions, assumptions and approvals (the model card and experiment log support this).
""")
    st.page_link("app_pages/10_Reports_Documentation.py", label="Open the model card and reports", icon=":material/description:")
