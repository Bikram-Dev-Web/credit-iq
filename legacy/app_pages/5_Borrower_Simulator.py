"""Page 5 — Borrower Risk Simulator: score one hypothetical applicant with the active model."""
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import evaluation as ev
from src import explainability as xai
from src import persistence as ps
from src import state, ui
from src.feature_engineering import DOMAIN_FEATURES
from src.financial_calculations import expected_loss
from src.portfolio_analysis import assign_risk_band, band_description

ss = st.session_state
ui.page_header("Borrower Risk Simulator", "Enter an applicant's details to see the model's estimated probability of default, "
               "the factors behind it and an illustrative expected loss. Decision support only — not an approval or denial.")
ui.require_model()

bundle = state.active_bundle()
model, schema = bundle.model, bundle.schema
threshold = state.active_threshold()
fin = state.fin()
exp = ss.experiment if not ss.use_loaded_bundle else None
exposure_col = bundle.metadata.get("exposure_column")
target_def = bundle.metadata.get("target", {}).get("definition") or "the dataset's default label"

# Feature ordering: most important first (model-internal importance needs no data).
try:
    imp, _ = xai.native_importance(model, bundle.transformed_mean)
    ranked = []
    for f in imp["feature"]:
        sources = DOMAIN_FEATURES[f][0] if f in DOMAIN_FEATURES else (f,)  # derived ratios rank their inputs
        ranked += [c for c in sources if c in schema and c not in ranked]
except (AttributeError, ValueError):
    ranked = []
ordered = ranked + [c for c in model.input_columns if c not in ranked]
TOP_N = 12

# ---------------------------------------------------------------- Starting point
c1, c2 = st.columns([2, 1])
with c1:
    starts = ["Typical applicant (training medians and most common values)"]
    if exp is not None:
        starts.append("A real held-out test applicant")
    start = st.radio("Start from", starts, horizontal=True)
row_values: dict = {}
observed = None
if start.startswith("A real") and exp is not None:
    Xt = exp.X_split("test")
    with c2:
        if "sim_row" not in ss or ss.sim_row >= len(Xt):
            ss.sim_row = 0
        b1, b2 = st.columns([2, 1])
        ss.sim_row = b1.number_input("Test row", 0, len(Xt) - 1, int(ss.sim_row))
        if b2.button("Random", icon=":material/casino:"):
            ss.sim_row = int(np.random.default_rng().integers(0, len(Xt)))
            st.rerun()
    row_values = Xt.iloc[int(ss.sim_row)].to_dict()
    observed = int(exp.y_split("test")[int(ss.sim_row)])
form_version = f"{start[:5]}_{ss.get('sim_row', 0)}"


def _default(col):
    info = schema[col]
    if row_values:
        v = row_values.get(col)
        return None if v is None or (isinstance(v, float) and np.isnan(v)) else v
    return info.get("median") if info["kind"] == "numeric" else info.get("mode")


def _input(col, container):
    info = schema[col]
    help_txt = f"Missing in {ui.pct(info['missing_rate'])} of training rows."
    if info["kind"] == "numeric":
        help_txt = (f"Training range (1st–99th pct): {info['p01']:,.4g} to {info['p99']:,.4g}; median {info['median']:,.4g}. "
                    + help_txt + " Leave blank for missing.")
        if col.startswith("DAYS_"):
            help_txt += " Days relative to the application (negative = in the past; 365243 is the dataset's placeholder)."
        d = _default(col)
        is_int = info.get("is_integer", False)
        val = container.number_input(col, value=None if d is None else (int(round(float(d))) if is_int else float(d)),
                                     step=1 if is_int else None, format=None if is_int else "%.6g", help=help_txt,
                                     key=f"in_{col}_{form_version}")
        return np.nan if val is None else val
    cats = list(info.get("categories", []))
    d = _default(col)
    opts = cats + ["(missing)"]
    if d is not None and str(d) not in cats:
        opts = [str(d)] + opts
    idx = opts.index(str(d)) if d is not None and str(d) in opts else len(opts) - 1
    val = container.selectbox(col, opts, index=idx, help=help_txt + f" {info.get('n_categories', len(cats))} distinct values in training.",
                              key=f"in_{col}_{form_version}")
    return np.nan if val == "(missing)" else val


# ---------------------------------------------------------------- Form
with st.form("applicant_form"):
    st.markdown(f"**Most influential inputs** ({min(TOP_N, len(ordered))} of {len(ordered)})")
    values = {}
    cols = st.columns(3)
    for i, col in enumerate(ordered[:TOP_N]):
        values[col] = _input(col, cols[i % 3])
    if len(ordered) > TOP_N:
        with st.expander(f"All other inputs ({len(ordered) - TOP_N}) — prefilled"):
            cols2 = st.columns(3)
            for i, col in enumerate(ordered[TOP_N:]):
                values[col] = _input(col, cols2[i % 3])
    st.markdown("**Loss assumptions for this applicant**")
    l1, l2, l3 = st.columns(3)
    lgd = ui.pct_input(l1, "LGD", fin["lgd"], step=1.0, digits=1, help="Loss given default: share of exposure lost after recoveries.")
    ccf = l2.number_input("Credit conversion factor", 0.0, 1.5, float(fin["ccf"]), 0.05, help="EAD = exposure × CCF (1.0 for a fully drawn term loan).")
    ead_manual = l3.number_input("Exposure if no loan-amount input", 0.0, value=float(fin["default_exposure"]), step=10_000.0,
                                 help=f"Used only when the model has no '{exposure_col or 'exposure'}' input.")
    submitted = st.form_submit_button("Assess applicant", type="primary", icon=":material/analytics:")

if submitted:
    X = pd.DataFrame([values])
    num_cols = [c for c in values if schema[c]["kind"] == "numeric"]
    if all(pd.isna(values[c]) for c in values):
        st.error("Every input is missing — enter at least some applicant details.")
        st.stop()
    out_of_range = [c for c in num_cols if not pd.isna(values[c]) and not (schema[c]["min"] <= float(values[c]) <= schema[c]["max"])]
    try:
        raw = float(model.predict_score(X)[0])
        pd_hat = float(model.calibrator.transform([raw])[0])
    except (ValueError, TypeError) as exc:
        st.error(f"The model could not score these inputs: {exc}")
        st.stop()
    exposure_val = values.get(exposure_col) if exposure_col else None
    ead = float(exposure_val) * ccf if exposure_val is not None and not pd.isna(exposure_val) else ead_manual * ccf
    el = expected_loss(pd_hat, lgd, ead)
    band = str(assign_risk_band([pd_hat], ss.band_cutoffs).iloc[0])
    flagged = pd_hat >= threshold
    try:
        expl, meta = xai.local_explanation(model, X, schema, bundle.transformed_mean, top=12)
    except Exception as exc:  # noqa: BLE001 — explanation failures must not hide the score
        expl, meta = None, {"error": f"{type(exc).__name__}: {exc}"}
    # Reliability: how did HELD-OUT test applicants with similar PDs actually behave? (The validation split
    # was used to fit the calibrator, so it would flatter the model here.)
    tab = ev.calibration_table(bundle.eval_test["y"], bundle.eval_test["pd"], 10)
    near = tab[(tab["pd_from"] <= pd_hat) & (pd_hat <= tab["pd_to"])]
    near = near.iloc[0].to_dict() if len(near) else (tab.iloc[-1].to_dict() if pd_hat > tab["pd_to"].max() else tab.iloc[0].to_dict())
    others = {}
    if exp is not None:
        for k, mdl_ in exp.models.items():
            others[mdl_.label] = float(mdl_.predict_pd(X)[0])
    ss.last_assessment = {
        "time": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"), "inputs": values, "raw": raw, "pd": pd_hat,
        "threshold": threshold, "band": band, "flagged": bool(flagged), "lgd": lgd, "ccf": ccf, "ead": ead, "el": el,
        "explanation": expl, "meta": meta, "reliability": near, "others": others, "out_of_range": out_of_range,
        "observed": observed, "model": model.label, "experiment_id": bundle.metadata.get("experiment_id"),
        "missing_cols": model.check_input(X),
        "blank_inputs": [c for c in values if pd.isna(values[c])],
    }
    try:
        ps.log_assessment({"experiment_id": bundle.metadata.get("experiment_id"), "model": model.key, "pd": pd_hat, "raw_score": raw,
                           "threshold": threshold, "risk_band": band, "outcome": "flagged for review" if flagged else "within threshold",
                           "ead": ead, "expected_loss": el, "input_hash": ps.hash_inputs(values)})
    except Exception as exc:  # noqa: BLE001
        st.warning(f"Assessment computed, but the audit record could not be written: {exc}")

a = ss.get("last_assessment")
if a and a.get("experiment_id") == bundle.metadata.get("experiment_id") and a.get("model") == model.label:
    st.divider()
    st.subheader("Assessment")
    calibrated = model.is_calibrated
    if not calibrated:
        ui.calibration_warning(bundle)
    ui.tiles([
        {"label": "Estimated probability of default" if calibrated else "Uncalibrated model score",
         "value": ui.pct(a["pd"], 2), "prov": "model",
         "help": (f"Calibrated probability of: {target_def}" if calibrated else
                  "Calibration was switched off, so this is a score, not a probability."),
         "sub": "calibrated on validation data" if calibrated else "not a probability of default"},
        {"label": "Raw model score", "value": ui.dec(a["raw"], 4), "prov": "model",
         "help": "The algorithm's uncalibrated output. With class re-weighting it overstates risk; it is a ranking score, not a PD."},
        {"label": "Review threshold", "value": ui.pct(a["threshold"], 2), "prov": "assumption",
         "help": "PD at or above which an application is flagged for human review (chosen on validation data or set in the Model Lab)."},
        {"label": "Risk band", "value": a["band"], "prov": "model", "help": "; ".join(band_description(ss.band_cutoffs)),
         "sub": "explanatory category, not a guarantee"},
        {"label": "Exposure at default", "value": ui.money(a["ead"]), "prov": "assumption", "sub": f"CCF {a['ccf']:.2f}"},
        {"label": "Illustrative 12-month expected loss", "value": ui.money(a["el"]), "prov": "simulated",
         "help": "PD × LGD × EAD, treating the model's PD as a 12-month probability (an assumption — see the target definition).",
         "sub": f"{ui.pct(a['pd'], 2)} × {ui.pct(a['lgd'], 0)} × {ui.money(a['ead'])}"},
    ])
    if a["flagged"]:
        ui.callout(f"<b>Flagged for review.</b> The estimated PD is at or above the {ui.pct(a['threshold'], 1)} threshold"
                   f"{' — elevated estimated risk' if a['band'] in ('Elevated', 'High') else ''}. A credit officer should review the "
                   "application with information the model does not see. This is not a denial.", "risk")
    else:
        ui.callout(f"<b>Within the selected threshold.</b> The estimated PD is below {ui.pct(a['threshold'], 1)}. This is not an approval: "
                   "affordability, verification and policy checks still apply.", "ok")
    rel = a["reliability"]
    unc = [f"Among held-out test applicants scored between {ui.pct(rel['pd_from'], 1)} and {ui.pct(rel['pd_to'], 1)} "
           f"(n = {int(rel['n']):,}), {ui.pct(rel['observed_rate'], 1)} actually defaulted "
           f"(95% interval {ui.pct(rel['ci_low'], 1)}–{ui.pct(rel['ci_high'], 1)})."]
    if len(a["others"]) > 1:
        vals = list(a["others"].values())
        unc.append(f"The trained models disagree within {ui.pct(min(vals), 1)}–{ui.pct(max(vals), 1)} for this applicant: "
                   + ", ".join(f"{k.split(' (')[0]} {ui.pct(v, 1)}" for k, v in a["others"].items()) + ".")
    if a["out_of_range"]:
        unc.append(f"Outside the training range: <b>{', '.join(a['out_of_range'])}</b>. The model is extrapolating; treat the score with extra caution.")
    for msg in a["missing_cols"]:
        unc.append(msg)
    blank = a.get("blank_inputs") or []
    if blank:
        unc.append(f"{len(blank)} of {len(a['inputs'])} inputs were left blank and are treated as missing (imputed, with a "
                   "missing-value indicator). The more inputs are missing, the less this score reflects this applicant.")
    if a.get("observed") is not None:
        unc.append(f"Historical outcome of this test applicant (observed label, not a prediction): "
                   f"<b>{'defaulted' if a['observed'] == 1 else 'did not default'}</b>.")
    ui.callout("<b>Uncertainty and caveats</b><br>" + "<br>".join(unc), "warn")

    st.subheader("Why this score")
    expl, meta = a["explanation"], a["meta"]
    if expl is None:
        st.error(f"An explanation could not be generated ({meta.get('error')}). The score above is unaffected.")
    else:
        e = expl.iloc[::-1]
        colors = [ui.DEFAULT_COLOR if v > 0 else "#1E4C8A" for v in e["contribution"]]
        labels = [f"{f}{' (derived)' if d else ''} = {ui.val(v)}" for f, v, d in zip(e["feature"], e["value"], e["derived"])]
        fig = go.Figure(go.Bar(x=e["contribution"], y=labels, orientation="h", marker_color=colors,
                               hovertemplate="%{y}<br>contribution %{x:+.4f}<extra></extra>"))
        fig.add_vline(x=0, line_color="#9AA8BC")
        fig.update_layout(title="Top contributions (red raises the score, blue lowers it)", xaxis_title=f"Contribution ({meta['units']})")
        ui.plot(fig, height=max(320, 30 * len(e) + 90))
        ui.note(f"Method: {meta['method']}. {meta.get('fallback_reason', '')} Contributions describe how this model uses the inputs "
                "relative to an average training applicant. They reflect statistical association in the training data, not causes "
                "of default, and are less stable when inputs are correlated (e.g. the three external scores).")

    st.subheader("Export")
    expl_records = a["explanation"].to_dict("records") if a["explanation"] is not None else []
    report = {
        "generated": a["time"], "disclaimer": "Decision support only. Not a credit approval or denial.",
        "model": a["model"], "experiment_id": a["experiment_id"], "target_definition": target_def,
        "inputs": a["inputs"], "raw_model_score": a["raw"], "calibrated_pd": a["pd"], "threshold": a["threshold"],
        "outcome": "flagged for review" if a["flagged"] else "within the selected threshold", "risk_band": a["band"],
        "risk_band_definition": band_description(ss.band_cutoffs),
        "loss_assumptions": {"lgd": a["lgd"], "ccf": a["ccf"], "ead": a["ead"]}, "expected_loss": a["el"],
        "explanation_method": a["meta"].get("method"), "explanation_units": a["meta"].get("units"),
        "top_contributions": expl_records, "validation_reliability": a["reliability"], "model_disagreement": a["others"],
        "caveats": {"out_of_training_range": a["out_of_range"], "missing_inputs": a["missing_cols"]},
    }
    c1, c2 = st.columns(2)
    with c1:
        ui.download("Assessment (JSON)", ps.to_json(report, indent=2), "borrower_assessment.json", "application/json")
    with c2:
        flat = {**{f"input_{k}": v for k, v in a["inputs"].items()}, "calibrated_pd": a["pd"], "raw_score": a["raw"],
                "threshold": a["threshold"], "risk_band": a["band"], "flagged_for_review": a["flagged"], "lgd": a["lgd"],
                "ead": a["ead"], "expected_loss": a["el"], "model": a["model"]}
        ui.download("Assessment (CSV)", pd.DataFrame([flat]).to_csv(index=False), "borrower_assessment.csv", "text/csv")
    ui.note("Each assessment adds one row to the local audit table (scores, band and a one-way hash of the inputs — the inputs "
            "themselves are not stored).")
else:
    ui.note(f"Risk bands: {'; '.join(band_description(ss.band_cutoffs))}. Edit the cutoffs in Portfolio Intelligence.")
