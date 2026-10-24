"""Page 4 — AI Model Lab: train, calibrate, validate, compare, export."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import config
from src import data_validation as dv
from src import evaluation as ev
from src import modeling as mdl
from src import persistence as ps
from src import state, ui
from src.calibration import CALIBRATION_METHODS

ss = st.session_state
ui.page_header("AI Model Lab", "Train three different algorithms on the training split, calibrate and choose a threshold on the "
               "validation split, and report final performance on the untouched test split.")


def _saved_models_section():
    with st.expander("Saved models and experiment history", expanded=not state.has_target() and not state.has_experiment()):
        bundles = ps.list_bundles()
        c1, c2 = st.columns([2, 1])
        with c1:
            if bundles:
                choice = ui.pick(st.selectbox, "Saved model bundles in models/", {b: b.name for b in bundles})
                if st.button("Load as the active model", icon=":material/upload:"):
                    try:
                        state.set_active_model(loaded_bundle=ps.load_bundle(choice))
                        ui.flash(f"Loaded {ss.loaded_bundle.model.label} (experiment {ss.loaded_bundle.metadata.get('experiment_id')}).")
                        st.rerun()
                    except (ValueError, OSError, EOFError, ModuleNotFoundError, AttributeError) as exc:
                        st.error(f"Could not load this bundle: {exc}")
            else:
                ui.note("No saved bundles yet. Train a model and use “Save bundle to models/”.")
            if ss.use_loaded_bundle and state.has_experiment():
                if st.button("Use this session's trained model instead"):
                    state.set_active_model(ss.active_model_key or ss.experiment.selected)
                    st.rerun()
        with c2:
            ui.note("Bundles contain the fitted pipeline, calibration and summary statistics — no training rows. "
                    "They are Python pickles: only load files you created.")
        hist = ps.list_experiments()
        if hist.empty:
            ui.note("No experiments recorded yet.")
        else:
            show = hist[["created_at", "id", "dataset_name", "is_synthetic", "target_column", "n_rows", "selected_model", "test_metrics", "artifact_path"]]
            ui.show_table(show, height=260)


if not state.has_target():
    ui.empty_state("Confirm the target first", "Load data and confirm the default label in the Data Center before training. "
                   "You can still load a saved model below.", "app_pages/2_Data_Center.py", "Open the Data Center")
    _saved_models_section()
    st.stop()

if not state.has_features():
    ui.callout("Predictors have not been confirmed yet. Review them in the Data Center, or accept the suggested set "
               "(identifiers, known post-outcome fields, dates and potentially protected columns excluded; the leakage "
               "guard then re-screens the rest on the training split).", "warn")
    if st.button("Use the suggested predictor set", type="primary"):
        df = ss.df
        dates = dv.detect_date_columns(df)
        date_col = dates[0] if dates else None
        roles = dv.suggest_feature_roles(df, ss.target.column, date_col)
        preds = roles.loc[roles["use"], "column"].tolist()
        excluded = {r.column: r.role for r in roles.itertuples() if not r.use and r.column != ss.target.column}
        exp_s = dv.suggest_exposure_columns(df)
        state.set_features(preds, excluded, date_col, exp_s[0] if exp_s else None,
                           roles.loc[roles["role"] == "sensitive", "column"].tolist(), roles)
        st.rerun()
    _saved_models_section()
    st.stop()

# ---------------------------------------------------------------- Configuration
n_labelled = ss.target.n_positive + ss.target.n_negative
st.subheader("Training configuration")
ui.note(f"{len(ss.predictors)} predictors · {n_labelled:,} labelled rows · boosting backend: "
        f"{mdl.BACKEND_LABELS[mdl.boosting_backend()]}")
if mdl.boosting_backend() == "sklearn_hgb":
    ui.callout("LightGBM and XGBoost are not installed, so the gradient-boosting model uses scikit-learn's "
               "HistGradientBoosting — a genuine gradient-boosting algorithm. Install <span class='ciq-code'>requirements-optional.txt</span> "
               "for LightGBM.", "warn")

with st.form("train_config"):
    c1, c2, c3 = st.columns(3)
    with c1:
        keys = ui.pick_many(st.multiselect, "Models", {k: mdl.model_label(k) for k in mdl.MODEL_KEYS}, list(mdl.MODEL_KEYS))
        sel_metric = ui.pick(st.selectbox, "Select the best model by (validation)",
                             {k: v[0] for k, v in mdl.SELECTION_METRICS.items()})
        thr_rule = ui.pick(st.selectbox, "Decision threshold rule (validation)", ev.THRESHOLD_RULES)
        manual_thr = st.slider("Manual threshold, PD % (if chosen)", 1.0, 99.0, 20.0, 0.5) / 100
    with c2:
        strategies = ["stratified"] + (["chronological"] if ss.date_col else [])
        split = ui.pick(st.selectbox, "Split strategy",
                        {s: {"stratified": "Stratified random", "chronological": f"Chronological by {ss.date_col}"}[s] for s in strategies},
                        help="Chronological: earliest rows train, later rows validate, latest rows test.")
        val_size = st.slider("Validation share", 0.10, 0.30, 0.20, 0.05)
        test_size = st.slider("Test share", 0.10, 0.30, 0.20, 0.05)
        floor = min(1_000, n_labelled)
        sample = st.number_input("Maximum labelled rows used (stratified sample)", floor, max(n_labelled, floor),
                                 min(config.DEFAULT_TRAIN_SAMPLE, n_labelled), step=5_000 if n_labelled > 10_000 else 100,
                                 help="Keeps training fast on a laptop. The full Home Credit file has ~307k rows.")
    with c3:
        cw = ui.pick(st.selectbox, "Class imbalance", {"balanced": "Re-weight classes (balanced)", "none": "No re-weighting"})
        cal = ui.pick(st.selectbox, "Probability calibration", {k: v.split(" (")[0] for k, v in CALIBRATION_METHODS.items()})
        seed = st.number_input("Random seed", 0, 1_000_000, config.DEFAULT_SEED)
        domain = st.checkbox("Credit ratio features", value=True, help="Adds ratios such as loan ÷ income and instalment ÷ income when the source columns are predictors.")
        tune = st.checkbox("Tune on validation", value=False, help="Tries three hyperparameter candidates per model and keeps the best by validation ROC-AUC. Slower.")
        guard = st.checkbox("Leakage guard", value=True,
                            help="Before fitting, drops any predictor that on its own separates the TRAINING labels almost "
                                 "perfectly (single-feature AUC ≥ 0.90). Uses training rows only.")
    with st.expander("Hyperparameters"):
        params = {}
        h1, h2, h3 = st.columns(3)
        with h1:
            st.markdown("**Logistic Regression**")
            params["logreg"] = {"C": st.number_input("Inverse regularisation C", 0.001, 100.0, 0.3, format="%.3f")}
        with h2:
            st.markdown("**Random Forest**")
            params["random_forest"] = {"n_estimators": st.number_input("Trees", 50, 1000, 200, 50),
                                       "max_depth": st.number_input("Max depth", 2, 40, 12),
                                       "min_samples_leaf": st.number_input("Min samples per leaf", 1, 500, 20)}
        with h3:
            st.markdown(f"**{mdl.model_label('gbm')}**")
            gp = mdl.default_params("gbm")
            backend = mdl.boosting_backend()
            n_key = "max_iter" if backend == "sklearn_hgb" else "n_estimators"
            params["gbm"] = {n_key: st.number_input("Boosting rounds", 50, 3000, int(gp[n_key]), 50),
                             "learning_rate": st.number_input("Learning rate", 0.005, 0.5, float(gp["learning_rate"]), format="%.3f")}
            if backend == "lightgbm":
                params["gbm"]["num_leaves"] = st.number_input("Leaves per tree", 4, 256, int(gp["num_leaves"]))
            elif backend == "xgboost":
                params["gbm"]["max_depth"] = st.number_input("Max depth ", 2, 12, int(gp["max_depth"]))
            else:
                params["gbm"]["max_leaf_nodes"] = st.number_input("Leaves per tree", 4, 256, int(gp["max_leaf_nodes"]))
    submitted = st.form_submit_button("Train models", type="primary", icon=":material/play_arrow:")

if submitted:
    if not keys:
        st.error("Select at least one model.")
    else:
        cfg = mdl.TrainingConfig(model_keys=keys, sample_size=int(sample), split_strategy=split, val_size=val_size,
                                 test_size=test_size, seed=int(seed), class_weight=cw, calibration=cal,
                                 threshold_rule=thr_rule, manual_threshold=manual_thr, selection_metric=sel_metric,
                                 tune=tune, use_domain_features=domain, params=params, leakage_guard=guard)
        if cal == "none":
            st.warning("Calibration is off: outputs will be model scores, not probabilities of default"
                       + (" — and class re-weighting inflates them." if cw == "balanced" else "."))
        bar = st.progress(0.0, text="Starting")
        with st.status("Training models", expanded=True) as status:
            try:
                result = mdl.run_experiment(
                    ss.df, ss.ds_info.to_dict(), ss.target, ss.predictors, cfg, date_col=ss.date_col,
                    exposure_col=ss.exposure_col, sensitive_cols=ss.sensitive_cols, excluded=ss.excluded,
                    progress=lambda f, msg: (bar.progress(f, text=msg), status.write(msg)))
                state.set_experiment(result)
                try:
                    ps.save_experiment(result)
                except Exception as exc:  # noqa: BLE001 — recording metadata must not lose the trained model
                    st.warning(f"Model trained, but the experiment record could not be saved: {exc}")
                status.update(label=f"Trained {len(result.models)} model(s) — selected {result.selected_model.label}", state="complete", expanded=False)
                ui.flash(f"Trained {len(result.models)} model(s) in experiment {result.experiment_id}. "
                         f"Best on validation: {result.selected_model.label}.")
                st.rerun()
            except (ValueError, MemoryError) as exc:
                status.update(label="Training stopped", state="error")
                st.error(str(exc))

exp = ss.experiment
if exp is None:
    _saved_models_section()
    st.stop()

# ---------------------------------------------------------------- Results
st.divider()
st.subheader("Results on held-out data")
prov = "synthetic" if exp.dataset.get("is_synthetic") else "measured"
for w in exp.warnings:
    st.warning(w)

options = list(exp.models)
active = ss.active_model_key if ss.active_model_key in options else exp.selected
choice = ui.pick(st.radio, "Active model (used by the simulator, portfolio, policy and monitoring pages)",
                 {k: exp.models[k].label + ("  — best on validation" if k == exp.selected else "") for k in options},
                 active, horizontal=True)
if choice != ss.active_model_key:
    state.set_active_model(choice)
    st.rerun()
key = choice
m_test = exp.metrics[key]["test"]
ui.tiles([
    {"label": "ROC-AUC", "value": ui.dec(m_test["roc_auc"], 3), "prov": prov, "help": ev.METRIC_HELP["roc_auc"],
     "sub": f"Gini {ui.dec(m_test['gini'], 3)}"},
    {"label": "PR-AUC (average precision)", "value": ui.dec(m_test["pr_auc"], 3), "prov": prov, "help": ev.METRIC_HELP["pr_auc"],
     "sub": f"random baseline {ui.dec(m_test['prevalence'], 3)}"},
    {"label": "Brier score", "value": ui.dec(m_test["brier"], 4), "prov": prov, "help": ev.METRIC_HELP["brier"]},
    {"label": "KS statistic", "value": ui.dec(m_test["ks"], 3), "prov": prov, "help": ev.METRIC_HELP["ks"]},
    {"label": "Lift in riskiest 10%", "value": f"{ui.dec(m_test['lift_top10'], 2)}×", "prov": prov, "help": ev.METRIC_HELP["lift_top10"]},
    {"label": "Test default rate", "value": ui.pct(m_test["prevalence"], 2), "prov": "observed", "help": ev.METRIC_HELP["prevalence"],
     "sub": f"{m_test['n']:,} test applications"},
])

comp = exp.comparison_table()
ui.show_table(comp.drop(columns=["key"]),
              dec_cols=["val_roc_auc", "val_pr_auc", "val_brier", "test_roc_auc", "test_gini", "test_pr_auc", "test_brier",
                        "test_log_loss", "test_ks", "train_roc_auc", "overfit_gap_auc", "fit_seconds"],
              pct_cols=["threshold", "test_recall", "test_precision", "test_specificity", "test_f1", "test_balanced_accuracy"])
ui.note("Validation columns were used to choose the model, calibrate it and set its threshold; test columns were computed "
        "once, afterwards. A large train-minus-test AUC gap indicates overfitting.")

tabs = ui.tabs(["Discrimination", "Calibration", "Threshold explorer", "Lift & KS", "Stability over time", "Training details", "Export"], key="tabs_model_lab")
y_test = exp.y_split("test")

with tabs[0]:
    c1, c2 = st.columns(2)
    fig_roc, fig_pr = go.Figure(), go.Figure()
    for k, model in exp.models.items():
        p = exp.pd_split("test", k)
        r = ev.roc_points(y_test, p)
        fig_roc.add_scatter(x=r["fpr"], y=r["tpr"], mode="lines", name=f"{model.label.split(' (')[0]} ({ui.dec(exp.metrics[k]['test']['roc_auc'], 3)})",
                            line=dict(color=ui.MODEL_COLORS[k], width=2.4 if k == key else 1.5))
        pr = ev.pr_points(y_test, p)
        fig_pr.add_scatter(x=pr["recall"], y=pr["precision"], mode="lines", name=model.label.split(" (")[0],
                           line=dict(color=ui.MODEL_COLORS[k], width=2.4 if k == key else 1.5))
    fig_roc.add_scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random", line=dict(color="#B8C4D4", dash="dot"))
    fig_roc.update_layout(title="ROC curve (test)", xaxis_title="False positive rate (good applicants flagged)",
                          yaxis_title="True positive rate (defaulters flagged)")
    fig_pr.add_hline(y=float(y_test.mean()), line_dash="dot", line_color="#B8C4D4", annotation_text="default rate")
    fig_pr.update_layout(title="Precision–recall curve (test)", xaxis_title="Recall", yaxis_title="Precision")
    with c1:
        ui.plot(fig_roc, height=400)
    with c2:
        ui.plot(fig_pr, height=400)

with tabs[1]:
    ui.note("A calibrated PD of 10% should mean that about 10% of such applicants default. The raw model score is not a "
            "probability when classes are re-weighted; the calibrator (fitted on validation) corrects it.")
    show_raw = st.checkbox("Also show the raw (uncalibrated) score of the active model")
    fig = go.Figure()
    rows = []
    for k, model in exp.models.items():
        tab = ev.calibration_table(y_test, exp.pd_split("test", k), 10)
        rows.append({"model": model.label, "ece_calibrated": ev.expected_calibration_error(tab),
                     "ece_raw_score": ev.expected_calibration_error(ev.calibration_table(y_test, exp.score_split("test", k), 10)),
                     "brier": exp.metrics[k]["test"]["brier"], "calibration": model.calibrator.describe()})
        fig.add_scatter(x=tab["mean_pd"], y=tab["observed_rate"], mode="lines+markers", name=model.label.split(" (")[0],
                        line=dict(color=ui.MODEL_COLORS[k]),
                        error_y=dict(type="data", symmetric=False, array=tab["ci_high"] - tab["observed_rate"],
                                     arrayminus=tab["observed_rate"] - tab["ci_low"], thickness=1) if k == key else None,
                        customdata=tab[["n"]], hovertemplate="Predicted %{x:.2%}<br>Observed %{y:.2%}<br>n=%{customdata[0]:,}<extra></extra>")
    if show_raw:
        tab = ev.calibration_table(y_test, exp.score_split("test", key), 10)
        fig.add_scatter(x=tab["mean_pd"], y=tab["observed_rate"], mode="lines+markers", name="Raw score (active)",
                        line=dict(color="#8A9BB3", dash="dash"))
    xs = [np.nanmax(np.asarray(t.x, dtype=float)) for t in fig.data if t.x is not None and len(t.x)]
    mx = float(max([0.05] + xs)) * 1.1
    fig.add_scatter(x=[0, mx], y=[0, mx], mode="lines", name="Perfect calibration", line=dict(color="#B8C4D4", dash="dot"))
    fig.update_layout(title="Calibration curve (test, 10 quantile bins)", xaxis=dict(title="Mean predicted PD", tickformat=".0%"),
                      yaxis=dict(title="Observed default rate", tickformat=".0%"))
    c1, c2 = st.columns([1.5, 1])
    with c1:
        ui.plot(fig, height=420)
    with c2:
        p = exp.pd_split("test", key)
        h = go.Figure()
        h.add_histogram(x=p[y_test == 0], name="Non-default", marker_color=ui.REPAID_COLOR, opacity=0.75, histnorm="percent", nbinsx=40)
        h.add_histogram(x=p[y_test == 1], name="Default", marker_color=ui.DEFAULT_COLOR, opacity=0.75, histnorm="percent", nbinsx=40)
        h.update_layout(barmode="overlay", title="PD distribution by observed outcome", xaxis_tickformat=".0%", yaxis_title="% of group")
        ui.plot(h, height=420)
    ui.show_table(pd.DataFrame(rows), dec_cols=["ece_calibrated", "ece_raw_score", "brier"])

with tabs[2]:
    c1, c2 = st.columns([1, 2])
    with c1:
        which = ui.pick(st.radio, "Evaluate on", {"validation": "Validation (sets the threshold)", "test": "Test (held-out, read-only)"},
                        "validation", horizontal=True)
        current = ss.threshold_override.get((exp.experiment_id, key), exp.thresholds[key])
        thr = st.slider("Decision threshold on " + ("calibrated PD" if exp.models[key].is_calibrated else "model score") + " (%)",
                        0.5, 95.0, float(min(95.0, max(0.5, round(current * 100 * 2) / 2))), 0.5, format="%.1f") / 100
        st.caption(f"Rule-based threshold from validation: {ui.pct(exp.thresholds[key], 2)} "
                   f"({ev.THRESHOLD_RULES[exp.config.threshold_rule]}). In use across the app: {ui.pct(current, 2)}.")
        if which == "validation":
            if st.button("Use this threshold across the app"):
                state.set_threshold(float(thr), exp.experiment_id, key)
                ui.flash(f"Review threshold set to {ui.pct(thr, 2)} for the simulator, portfolio and policy pages.")
                st.rerun()
        else:
            ui.note("Thresholds are chosen on validation data only. Switch to the validation view to change it; the test "
                    "view shows what the chosen threshold does on unseen applicants.")
    y_s, p_s = exp.y_split(which), exp.pd_split(which, key)
    m = ev.classification_metrics(y_s, p_s, thr)
    with c2:
        ui.tiles([
            {"label": "Applicants flagged", "value": f"{m['flagged']:,}", "sub": f"{ui.pct(m['flag_rate'])} of {m['n']:,}", "prov": "model",
             "help": ev.METRIC_HELP["flag_rate"]},
            {"label": "Recall (defaulters caught)", "value": ui.pct(m["recall"]), "prov": prov, "help": ev.METRIC_HELP["recall"]},
            {"label": "Precision", "value": ui.pct(m["precision"]), "prov": prov, "help": ev.METRIC_HELP["precision"]},
            {"label": "Specificity", "value": ui.pct(m["specificity"]), "prov": prov, "help": ev.METRIC_HELP["specificity"]},
            {"label": "Balanced accuracy", "value": ui.pct(m["balanced_accuracy"]), "prov": prov, "help": ev.METRIC_HELP["balanced_accuracy"]},
        ])
    c3, c4 = st.columns([1, 1.6])
    with c3:
        z = [[m["tn"], m["fp"]], [m["fn"], m["tp"]]]
        labels = [["True negative<br>repaid, not flagged", "False positive<br>repaid, flagged"],
                  ["False negative<br>defaulted, not flagged", "True positive<br>defaulted, flagged"]]
        fig = go.Figure(go.Heatmap(z=np.log1p(z), x=["Predicted: within threshold", "Predicted: flagged"], y=["Observed: repaid", "Observed: defaulted"],
                                   colorscale=[[0, "#F3F6FA"], [1, "#9DBBE3"]], showscale=False, xgap=3, ygap=3,
                                   text=[[f"{labels[i][j]}<br><b>{z[i][j]:,}</b>" for j in range(2)] for i in range(2)],
                                   texttemplate="%{text}", textfont=dict(size=13, color=ui.INK), hoverinfo="skip"))
        fig.update_layout(title="Confusion matrix", yaxis=dict(autorange="reversed"))
        ui.plot(fig, height=360)
        ui.note("Rows are observed historical outcomes; columns are model decisions at this threshold.")
    with c4:
        sw = ev.threshold_sweep(y_s, p_s, np.round(np.linspace(0.01, 0.95, 95), 3))
        fig = go.Figure()
        for col, color, name in (("recall", "#A23B3B", "Recall"), ("precision", "#1E4C8A", "Precision"),
                                 ("specificity", "#2F6B5E", "Specificity"), ("flag_rate", "#8A9BB3", "Share flagged")):
            fig.add_scatter(x=sw["threshold"], y=sw[col], mode="lines", name=name, line=dict(color=color))
        fig.add_vline(x=thr, line_color=ui.INK, line_dash="dash")
        fig.update_layout(title="Metrics across thresholds", xaxis=dict(title="Threshold (PD)", tickformat=".0%"),
                          yaxis=dict(tickformat=".0%"))
        ui.plot(fig, height=360)

with tabs[3]:
    dt = ev.decile_table(y_test, exp.pd_split("test", key), 10)
    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure()
        fig.add_bar(x=dt["risk_group"], y=dt["default_rate"], name="Observed default rate", marker_color="#1E4C8A")
        fig.add_scatter(x=dt["risk_group"], y=dt["mean_pd"], name="Mean predicted PD", mode="lines+markers", line_color="#C2873A")
        fig.update_layout(title="Observed default rate by risk decile (1 = riskiest)", xaxis=dict(title="Risk decile", dtick=1),
                          yaxis_tickformat=".0%")
        ui.plot(fig)
    with c2:
        p = exp.pd_split("test", key)
        xs = np.sort(p)
        cdf_bad = np.searchsorted(np.sort(p[y_test == 1]), xs, side="right") / max((y_test == 1).sum(), 1)
        cdf_good = np.searchsorted(np.sort(p[y_test == 0]), xs, side="right") / max((y_test == 0).sum(), 1)
        idx = np.unique(np.linspace(0, len(xs) - 1, 300).astype(int))
        ks, ks_t = ev.ks_statistic(y_test, p)
        fig = go.Figure()
        fig.add_scatter(x=xs[idx], y=cdf_good[idx], name="Repaid", line_color=ui.REPAID_COLOR)
        fig.add_scatter(x=xs[idx], y=cdf_bad[idx], name="Defaulted", line_color=ui.DEFAULT_COLOR)
        fig.update_layout(title=f"KS: cumulative PD distributions (KS = {ui.dec(ks, 3)})",
                          xaxis=dict(title="PD", tickformat=".0%"), yaxis=dict(title="Cumulative share", tickformat=".0%"))
        ui.plot(fig)
    ui.show_table(dt, pct_cols=["min_pd", "max_pd", "mean_pd", "default_rate", "cum_capture"], dec_cols=["lift", "cum_lift"], int_cols=["n", "defaults"])

with tabs[4]:
    if not exp.date_col:
        ui.empty_state("No date column", "Select an application date in the Data Center to evaluate performance by period.")
    else:
        bp = ev.metrics_by_period(y_test, exp.pd_split("test", key), exp.frame_split("test")[exp.date_col], exp.thresholds[key], "Q")
        if bp.empty:
            ui.note("No valid dates in the test split.")
        else:
            fig = go.Figure()
            fig.add_scatter(x=bp["period"], y=bp["observed_default_rate"], name="Observed default rate", line_color=ui.DEFAULT_COLOR)
            fig.add_scatter(x=bp["period"], y=bp["mean_pd"], name="Mean predicted PD", line_color="#C2873A")
            fig.add_scatter(x=bp["period"], y=bp["roc_auc"], name="ROC-AUC (right axis)", yaxis="y2", line=dict(color="#1E4C8A", dash="dot"))
            fig.update_layout(title="Test-set performance by quarter", yaxis=dict(tickformat=".0%"),
                              yaxis2=dict(overlaying="y", side="right", showgrid=False, range=[0.5, 1]))
            ui.plot(fig)
            ui.show_table(bp, pct_cols=["observed_default_rate", "mean_pd", "recall", "precision"], dec_cols=["roc_auc", "brier"], int_cols=["n"])
            if exp.split.strategy == "stratified":
                ui.note("With a random split, every quarter appears in training too, so this is not a forward-looking test. "
                        "Use the chronological split for that.")

with tabs[5]:
    sd = exp.split.to_dict()
    ui.tiles([{"label": k.capitalize(), "value": f"{v:,}", "sub": " to ".join(sd["periods"][k]) if k in sd["periods"] else None}
              for k, v in sd["sizes"].items()])
    for n in sd["notes"]:
        ui.note(n)
    st.markdown("**Fitted models**")
    for k, model in exp.models.items():
        st.markdown(f"- **{model.label}** — {model.fit_seconds:.1f}s; parameters `{model.params}`; calibration: {model.calibrator.describe()}; "
                    f"derived features: {', '.join(model.derived_features()) or 'none'}")
    if exp.tuning:
        st.markdown("**Validation tuning trials**")
        for k, t in exp.tuning.items():
            st.markdown(f"*{exp.models[k].label}* — chosen `{t['chosen']}`")
            ui.show_table(pd.DataFrame(t["trials"]), dec_cols=["val_roc_auc"])
    with st.expander("Full configuration"):
        st.json(exp.config.to_dict())
    with st.expander(f"Predictors ({len(exp.predictors)}) and exclusions ({len(exp.excluded)})"):
        st.write(exp.predictors)
        st.write(exp.excluded)

with tabs[6]:
    meta = exp.metadata()
    c1, c2, c3 = st.columns(3)
    with c1:
        ui.download("Metrics (JSON)", ps.to_json(exp.metrics, indent=2), f"metrics_{exp.experiment_id}.json", "application/json")
        ui.download("Model comparison (CSV)", comp.to_csv(index=False), f"model_comparison_{exp.experiment_id}.csv", "text/csv")
    with c2:
        ui.download("Test predictions with observed labels (CSV)", exp.test_predictions().to_csv(index=False),
                    f"test_predictions_{exp.experiment_id}.csv", "text/csv")
        ui.download("Experiment metadata (JSON)", ps.to_json(meta, indent=2), f"experiment_{exp.experiment_id}.json", "application/json")
    with c3:
        bundle = state.experiment_bundle()  # this session's model, not a bundle loaded from disk
        name = f"creditiq_{exp.experiment_id}_{key}"
        if st.button("Save bundle to models/", icon=":material/save:"):
            path = ps.save_bundle(bundle, name)
            ps.update_experiment_artifact(exp.experiment_id, str(path))
            st.success(f"Saved to {path}")
        if st.toggle("Prepare model bundle for download"):
            ui.download("Download model bundle (.joblib)", ps.bundle_bytes(bundle), f"{name}.joblib", "application/octet-stream")

_saved_models_section()
