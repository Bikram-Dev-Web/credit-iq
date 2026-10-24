"""Page 7 — Financial Scenario Lab: lending economics, stress testing, policy comparison, loan NPV."""
from dataclasses import replace

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import decision_engine as de
from src import financial_calculations as fc
from src import portfolio_analysis as pa
from src import state, ui

ss = st.session_state
ui.page_header("Financial Scenario Lab", "Turn default probabilities into money: expected loss, simplified contribution, "
               "break-even pricing, stress scenarios, lending-policy trade-offs and loan NPV.")
fin = state.fin()
exp = ss.experiment
key = (ss.active_model_key or exp.selected) if exp is not None else None
port = None
if exp is not None:
    ui.calibration_warning(exp.models.get(key))
    port = pa.build_portfolio(exp.frame_split("test"), exp.y_split("test"), exp.pd_split("test", key), exp.exposure_col,
                              fin["lgd"], fin["ccf"], fin["default_exposure"], ss.band_cutoffs)

with st.expander("Shared assumptions (used by every tab)", expanded=False):
    c1, c2, c3, c4 = st.columns(4)
    fin["annual_rate"] = ui.pct_input(c1, "Annual lending rate", fin["annual_rate"], 200.0, 0.25)
    fin["funding_rate"] = ui.pct_input(c2, "Annual funding rate", fin["funding_rate"], 200.0, 0.25)
    fin["opex_per_loan"] = c3.number_input("Operating cost per loan per year", 0.0, value=float(fin["opex_per_loan"]), step=100.0,
                                           help="Annual servicing/operating cost of one loan; multiplied by the horizon.")
    fin["lgd"] = ui.pct_input(c4, "LGD", fin["lgd"], step=1.0, digits=1)
    c5, c6, c7, _ = st.columns(4)
    fin["horizon_years"] = c5.number_input("Horizon (years)", 0.25, 10.0, float(fin["horizon_years"]), 0.25,
                                           help="Interest, funding, operating cost and expected loss all use this horizon. The annual "
                                                "PD is converted to the horizon as 1 − (1 − PD)^h (constant annual hazard).")
    fin["default_exposure"] = c6.number_input("Default exposure per loan", 0.0, value=float(fin["default_exposure"]), step=10_000.0)
    fin["currency"] = c7.text_input("Currency symbol for display", fin.get("currency", ""), max_chars=4)
    ui.note("PD is the model's probability of the dataset's target event and is treated here as a <b>12-month (annual)</b> PD; over a "
            "horizon of h years it becomes 1 − (1 − PD)^h. That mapping is an assumption — the Home Credit target, for example, "
            "concerns payment difficulty on early instalments, not a 12-month default.")

tab_econ, tab_stress, tab_policy, tab_npv = ui.tabs(["Lending economics", "Stress testing", "Policy comparison", "Loan NPV"], key="tabs_financial_scenarios")

# ---------------------------------------------------------------- Economics
with tab_econ:
    defaults = {"n_loans": 1000, "avg_exposure": fin["default_exposure"], "pd": 0.08}
    if port is not None:
        k = pa.portfolio_kpis(port)
        defaults = {"n_loans": int(k["n_loans"]), "avg_exposure": float(k["total_ead"] / k["n_loans"]),
                    "pd": float(k["ead_weighted_pd"])}
        ui.note(f"Prefilled from the held-out test portfolio: {k['n_loans']:,} loans, average EAD {ui.money(defaults['avg_exposure'])}, "
                f"EAD-weighted PD {ui.pct(defaults['pd'], 2)} (model output). Edit freely.")
    c1, c2, c3, c4, c5 = st.columns(5)
    n_loans = c1.number_input("Number of loans", 0, 10_000_000, defaults["n_loans"], 100)
    avg_exp = c2.number_input("Average exposure", 0.0, value=defaults["avg_exposure"], step=10_000.0)
    pd_in = ui.pct_input(c3, "Annual PD", defaults["pd"], step=0.5, digits=3, help="12-month probability of default.")
    fee = ui.pct_input(c4, "Upfront fee", 0.0, 20.0, 0.25, help="Fee as a share of exposure, received at origination.")
    k_int = ui.pct_input(c5, "Interest paid by defaulters", 0.0, step=5.0, digits=0,
                         help="Share of contractual interest still received from loans that default (0 = none).")
    a = fc.LendingAssumptions(n_loans=int(n_loans), avg_exposure=avg_exp, pd=pd_in, lgd=fin["lgd"], annual_rate=fin["annual_rate"],
                              funding_rate=fin["funding_rate"], opex_per_loan=fin["opex_per_loan"], horizon_years=fin["horizon_years"],
                              upfront_fee_rate=fee, interest_recovery_on_default=k_int)
    try:
        r = fc.lending_contribution(a)
    except ValueError as exc:
        st.error(str(exc))
        st.stop()
    ss.contribution_result = {"assumptions": a.to_dict(), **r}
    h_txt = f"{fin['horizon_years']:g}-year"
    ui.tiles([
        {"label": f"PD over the {h_txt} horizon", "value": ui.pct(r["pd_over_horizon"], 2), "prov": "simulated",
         "help": "PD_h = 1 − (1 − annual PD)^h", "sub": f"from {ui.pct(pd_in, 2)} a year"},
        {"label": "Expected interest income", "value": ui.money(r["expected_interest_income"]), "prov": "simulated",
         "help": "rate × exposure × h × [(1 − PD_h) + PD_h × share of interest collected from defaulters]"},
        {"label": "Funding cost", "value": ui.money(r["funding_cost"]), "prov": "simulated", "help": "funding rate × exposure × h"},
        {"label": "Expected credit loss", "value": ui.money(r["expected_credit_loss"]), "prov": "simulated",
         "help": "loans × PD_h × LGD × EAD", "sub": f"{ui.pct(r['el_rate'], 2)} of exposure"},
        {"label": "Operating cost", "value": ui.money(r["operating_cost"]), "prov": "assumption",
         "help": "loans × operating cost per loan per year × h"},
        {"label": "Expected contribution", "value": ui.money(r["expected_contribution"]), "prov": "simulated",
         "sub": f"{ui.pct(r['contribution_margin_on_exposure'], 2)} of exposure"},
        {"label": "Break-even annual PD", "value": ui.pct(r["break_even_pd"], 2), "prov": "simulated",
         "help": "Annual PD at which expected contribution is zero, holding everything else fixed (n/a = no break-even PD between 0% and 100%)."},
        {"label": "Break-even lending rate", "value": ui.pct(r["break_even_rate"], 2), "prov": "simulated",
         "help": "Minimum annual rate that covers funding, expected loss and operating cost at this PD (risk-based pricing floor)."},
    ])
    c1, c2 = st.columns(2)
    with c1:
        wf = go.Figure(go.Waterfall(
            x=["Interest income", "Fees", "Funding cost", "Expected credit loss", "Operating cost", "Contribution"],
            measure=["relative", "relative", "relative", "relative", "relative", "total"],
            y=[r["expected_interest_income"], r["fee_income"], -r["funding_cost"], -r["expected_credit_loss"], -r["operating_cost"], 0],
            increasing=dict(marker_color="#2F6B5E"), decreasing=dict(marker_color="#A23B3B"), totals=dict(marker_color="#1E4C8A"),
            connector=dict(line=dict(color="#C9D3E0"))))
        wf.update_layout(title=f"From interest income to expected contribution ({h_txt} horizon)", yaxis_title=ui.amount_label("Amount"),
                         showlegend=False)
        ui.plot(wf, height=380)
    with c2:
        sens = fc.sensitivity_analysis(a, 0.20)
        fig = go.Figure()
        fig.add_bar(y=sens["driver"], x=sens["change_if_driver_down"], orientation="h", name="Driver −20%", marker_color="#8A9BB3")
        fig.add_bar(y=sens["driver"], x=sens["change_if_driver_up"], orientation="h", name="Driver +20%", marker_color="#1E4C8A")
        fig.update_layout(title="Sensitivity of contribution to ±20% in each driver", barmode="overlay", xaxis_title=ui.amount_label("Change in contribution"))
        ui.plot(fig, height=380)
    pds = np.linspace(0.0, max(0.4, pd_in * 2), 41)
    rates = np.linspace(max(0.0, fin["funding_rate"]), fin["annual_rate"] * 1.8 + 0.02, 31)
    grid = np.array([[fc.lending_contribution(replace(a, pd=float(p_), annual_rate=float(r_)))["contribution_margin_on_exposure"]
                      for p_ in pds] for r_ in rates])
    fig = go.Figure(go.Heatmap(z=grid, x=pds, y=rates, zmid=0, colorscale=[[0, "#A23B3B"], [0.5, "#F7F7F7"], [1, "#2F6B5E"]],
                               colorbar=dict(tickformat=".0%", title="Margin"),
                               hovertemplate="PD %{x:.1%}<br>Rate %{y:.1%}<br>Contribution / exposure %{z:.2%}<extra></extra>"))
    fig.add_scatter(x=[pd_in], y=[fin["annual_rate"]], mode="markers", marker=dict(color=ui.INK, size=10, symbol="x"), name="Current inputs")
    fig.update_layout(title="Contribution margin by PD and lending rate (white = break-even)", xaxis=dict(title="Annual PD", tickformat=".0%"),
                      yaxis=dict(title="Annual lending rate", tickformat=".0%"))
    ui.plot(fig, height=420)
    if port is not None:
        st.markdown("**Risk-based pricing floor by risk band** (test portfolio)")
        bands = pa.band_summary(port)
        rows = []
        for b in bands.itertuples():
            if b.loans == 0:
                continue
            ab = replace(a, pd=float(b.avg_pd), avg_exposure=float(b.ead / b.loans))
            rows.append({"risk_band": str(b.risk_band), "loans": b.loans, "avg_pd": b.avg_pd, "observed_default_rate": b.observed_default_rate,
                         "break_even_rate": fc.break_even_rate(ab), "contribution_at_current_rate": fc.lending_contribution(ab)["contribution_margin_on_exposure"]})
        ui.show_table(pd.DataFrame(rows), pct_cols=["avg_pd", "observed_default_rate", "break_even_rate", "contribution_at_current_rate"], int_cols=["loans"])
    ui.callout("<b>What this calculation leaves out.</b> One-period view: no amortisation, prepayment, discounting, taxes, capital "
               "charges or recovery timing (see the Loan NPV tab for a cash-flow view). Principal repayment is a return of capital, "
               "not income; credit loss on principal enters once, via expected credit loss. It is a simplified contribution, not "
               "accounting profit.", "warn")

# ---------------------------------------------------------------- Stress
with tab_stress:
    ui.note("Scenarios scale each loan's PD, shift LGD and scale exposure. PD and LGD are capped at 100%. All multipliers are "
            "illustrative assumptions unless you calibrate them to evidence (e.g. historical downturn default rates).")
    sc_df = st.data_editor(pd.DataFrame(fc.DEFAULT_SCENARIOS), num_rows="dynamic", hide_index=True, key="scenarios",
                           column_config={"pd_multiplier": st.column_config.NumberColumn("PD ×", min_value=0.0, step=0.1),
                                          "lgd_shift": st.column_config.NumberColumn("LGD + (absolute)", min_value=-1.0, max_value=1.0, step=0.05),
                                          "ead_multiplier": st.column_config.NumberColumn("EAD ×", min_value=0.0, step=0.05)})
    sc_df = sc_df.dropna()
    if port is not None:
        base_pd, base_ead = port["pd"].to_numpy(), port["ead"].to_numpy()
        ui.note(f"Applied loan by loan to the {len(port):,}-loan held-out test portfolio (model PDs).")
    else:
        base_pd = np.full(1000, 0.08)
        base_ead = np.full(1000, fin["default_exposure"])
        ui.note("No trained model in this session: applied to an illustrative book of 1,000 identical loans with PD 8%.")
    try:
        stress = fc.stress_test(base_pd, fin["lgd"], base_ead, sc_df.to_dict("records"), fin["annual_rate"], fin["funding_rate"],
                                fin["opex_per_loan"], fin["horizon_years"])
    except (ValueError, KeyError) as exc:
        st.error(f"Check the scenario table: {exc}")
        st.stop()
    ss.scenario_table = stress
    base_row = stress[stress["scenario"].str.lower() == "base"]
    if len(base_row):
        stress["el_change_vs_base"] = stress["expected_loss"] - float(base_row["expected_loss"].iloc[0])
        stress["contribution_change_vs_base"] = stress["expected_contribution"] - float(base_row["expected_contribution"].iloc[0])
    fig = go.Figure()
    fig.add_bar(x=stress["scenario"], y=stress["expected_loss"], name="Expected loss", marker_color="#A23B3B")
    fig.add_bar(x=stress["scenario"], y=stress["expected_contribution"], name="Expected contribution", marker_color="#1E4C8A")
    fig.update_layout(title=f"Expected loss and contribution by scenario over {fin['horizon_years']:g} year(s) (illustrative)",
                      barmode="group", yaxis_title=ui.amount_label("Amount"))
    ui.plot(fig)
    ui.show_table(stress, pct_cols=["avg_pd", "avg_pd_over_horizon", "ead_weighted_pd", "avg_lgd", "el_rate"],
                  money_cols=["total_ead", "expected_loss", "expected_interest_income", "funding_cost", "operating_cost",
                              "expected_contribution", "el_change_vs_base", "contribution_change_vs_base"], int_cols=["n_loans"])
    ui.download("Scenario results (CSV)", stress.to_csv(index=False), "stress_scenarios.csv", "text/csv")

# ---------------------------------------------------------------- Policies
with tab_policy:
    if exp is None:
        ui.empty_state("Needs a trained model", "Policies are compared on held-out applicants scored by a model trained in this session.",
                       "app_pages/4_Model_Lab.py", "Open the AI Model Lab")
    else:
        ui.note("Flagged applicants are referred for review and, in this simplified matrix, not booked. Anything tuned (the cost-minimising "
                "threshold) is tuned on the validation split; all results below are on the test split.")
        c1, c2, c3, c4 = st.columns(4)
        mode = ui.pick(c1.radio, "Cost of errors", {"per_loan": "Scaled by each loan's EAD", "fixed": "Fixed amounts"})
        if mode == "per_loan":
            lost_margin = ui.pct_input(c2, "Lost margin on a rejected good applicant", max(fin["annual_rate"] - fin["funding_rate"], 0.0),
                                       step=0.5, help="Opportunity cost as a share of the loan's EAD (default: lending rate − funding rate).")
            c3.markdown(f"<div class='ciq-note' style='margin-top:1.9rem'>Missed default costs LGD × EAD = {ui.pct(fin['lgd'], 0)} × EAD "
                        "(LGD from the shared assumptions).</div>", unsafe_allow_html=True)
            costs = de.CostAssumptions(mode="per_loan", lgd=fin["lgd"], lost_margin_rate=lost_margin)
        else:
            fn_cost = c2.number_input("Missed default (approved defaulter)", 0.0, value=200_000.0, step=10_000.0)
            fp_cost = c3.number_input("Rejected good applicant", 0.0, value=40_000.0, step=5_000.0)
            costs = de.CostAssumptions(mode="fixed", missed_default_cost=fn_cost, rejected_good_cost=fp_cost, lgd=fin["lgd"])
        costs.review_cost = c4.number_input("Review cost per flagged application", 0.0, value=500.0, step=50.0)

        def _ead(split):
            f = exp.frame_split(split)
            if exp.exposure_col:
                e = pd.to_numeric(f[exp.exposure_col], errors="coerce")
                return (e.fillna(e.median()).clip(lower=0) * fin["ccf"]).to_numpy()
            return np.full(len(f), fin["default_exposure"] * fin["ccf"])

        thr = state.experiment_threshold()
        table, extra = de.compare_policies(exp.y_split("validation"), exp.pd_split("validation", key), _ead("validation"),
                                           exp.y_split("test"), exp.pd_split("test", key), _ead("test"), thr, costs,
                                           fin["annual_rate"], fin["funding_rate"], fin["opex_per_loan"], fin["horizon_years"])
        ss.policy_table = table
        best = table.loc[table["total_cost"].idxmin(), "policy"]
        ui.tiles([
            {"label": "Lowest total declared cost", "value": {"Baseline": "Approve all", "Fixed PD threshold": "Fixed threshold",
                                                               "Cost-minimising threshold": "Cost-minimising",
                                                               "Expected-contribution rule": "Contribution rule"}.get(
                best.split(" (")[0].split(" —")[0], best.split(" (")[0]), "prov": "simulated", "sub": "under these assumptions only"},
            {"label": "Cost-minimising threshold", "value": ui.pct(extra["cost_optimal_threshold"], 1), "prov": "simulated",
             "sub": "chosen on validation"},
            {"label": "Current fixed threshold", "value": ui.pct(thr, 1), "prov": "assumption"},
        ])
        show = table.rename(columns={"realized_contribution_approved_book": "back-tested_contribution"})
        ui.show_table(show, pct_cols=["approval_rate", "default_rate_among_approved"],
                      money_cols=["missed_default_cost", "rejected_good_cost", "review_cost_total", "total_cost", "cost_per_applicant",
                                  "back-tested_contribution"], int_cols=["tp", "fp", "tn", "fn", "flagged"])
        c1, c2 = st.columns(2)
        with c1:
            c_fn, c_fp = de.cost_vectors(costs, _ead("test"), len(exp.y_split("test")))
            curve = de.cost_curve(exp.y_split("test"), exp.pd_split("test", key), c_fn, c_fp, costs.review_cost,
                                  np.round(np.linspace(0, 0.8, 81), 3))
            fig = go.Figure()
            fig.add_scatter(x=curve["threshold"], y=curve["missed_default_cost"], name="Missed defaults", stackgroup="c", line_color="#A23B3B")
            fig.add_scatter(x=curve["threshold"], y=curve["rejected_good_cost"], name="Rejected good applicants", stackgroup="c", line_color="#C2873A")
            fig.add_scatter(x=curve["threshold"], y=curve["review_cost_total"], name="Review", stackgroup="c", line_color="#8A9BB3")
            fig.add_vline(x=extra["cost_optimal_threshold"], line_dash="dash", line_color=ui.INK, annotation_text="cost-minimising (val)")
            fig.add_vline(x=thr, line_dash="dot", line_color="#1E4C8A", annotation_text="fixed", annotation_position="bottom right")
            fig.update_layout(title="Total declared cost by threshold (test)", xaxis=dict(title="PD threshold", tickformat=".0%"),
                              yaxis_title=ui.amount_label("Cost"))
            ui.plot(fig, height=380)
        with c2:
            fig = go.Figure()
            names = [p.split(" (")[0].replace("Baseline — ", "") for p in table["policy"]]
            fig.add_bar(x=names, y=table["realized_contribution_approved_book"], marker_color="#1E4C8A", name="Back-tested contribution")
            fig.update_layout(title="Back-tested contribution of the approved book (test)", yaxis_title=ui.amount_label("Contribution"))
            ui.plot(fig, height=380)
        ui.callout("<b>Read with care.</b> A missed default does not cost the full loan amount (recoveries reduce it), and a rejected "
                   "applicant is an opportunity cost, not a booked loss — and may not have taken the loan anyway. The back-tested "
                   "contribution uses observed outcomes with the assumed LGD and rates. The preferred policy depends on the lender's "
                   "objectives, risk appetite, capacity to review and the reliability of these estimates; no threshold is universally optimal.", "warn")
        ui.download("Policy comparison (CSV)", table.to_csv(index=False), "policy_comparison.csv", "text/csv")

# ---------------------------------------------------------------- NPV
with tab_npv:
    ui.note("Cash-flow view of one amortising loan with a constant monthly default hazard implied by the annual PD. On default the "
            "lender recovers (1 − LGD) × outstanding balance after a lag. No prepayment.")
    d_pd = float(pa.portfolio_kpis(port)["avg_pd"]) if port is not None else 0.08
    d_principal = float(port["ead"].median()) if port is not None else fin["default_exposure"]
    c1, c2, c3, c4, c5 = st.columns(5)
    principal = c1.number_input("Principal", 1.0, value=round(d_principal, -3) or 100_000.0, step=10_000.0)
    rate = ui.pct_input(c2, "Annual rate", fin["annual_rate"], 200.0, 0.25, key="npv_rate")
    term = c3.number_input("Term (months)", 1, 480, 36)
    apd = ui.pct_input(c4, "Annual PD for this loan", d_pd, step=0.5, digits=3)
    disc = ui.pct_input(c5, "Discount rate", fin["funding_rate"], 200.0, 0.25, help="Lender's cost of funds or required return.")
    c6, c7, c8, _ = st.columns([1, 1, 1, 2])
    lag = c6.number_input("Recovery lag (months)", 0, 60, 6)
    fee_amt = c7.number_input("Upfront fee", 0.0, value=0.0, step=1_000.0)
    serv = c8.number_input("Servicing cost / month", 0.0, value=float(fin["opex_per_loan"]) / 12, step=10.0)
    try:
        res = fc.loan_npv(principal, rate, int(term), apd, fin["lgd"], disc, int(lag), fee_amt, serv)
    except ValueError as exc:
        st.error(str(exc))
        st.stop()
    ui.tiles([
        {"label": "Monthly instalment", "value": ui.money(res["monthly_payment"], compact=False), "prov": "simulated"},
        {"label": "NPV if no default", "value": ui.money(res["npv_contractual"]), "prov": "simulated"},
        {"label": "Risk-adjusted NPV", "value": ui.money(res["npv_risk_adjusted"]), "prov": "simulated"},
        {"label": "PV of credit cost", "value": ui.money(res["pv_credit_cost"]), "prov": "simulated",
         "help": "Contractual NPV − risk-adjusted NPV: lost principal and interest net of discounted recoveries."},
        {"label": "Lifetime default probability", "value": ui.pct(res["lifetime_pd"], 1), "prov": "simulated",
         "sub": f"from {ui.pct(apd, 2)} a year"},
        {"label": "Expected annual yield", "value": ui.pct(res["expected_annual_yield"], 2), "prov": "simulated",
         "sub": f"contractual {ui.pct(res['contractual_annual_yield'], 2)}"},
    ])
    cf = res["cash_flows"]
    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure()
        cf1 = cf[cf["month"] > 0]
        fig.add_bar(x=cf1["month"], y=cf1["contractual_cash_flow"], name="Contractual", marker_color="#D5DEEA")
        fig.add_bar(x=cf1["month"], y=cf1["expected_payments"], name="Expected payments", marker_color="#1E4C8A")
        fig.add_bar(x=cf1["month"], y=cf1["expected_recoveries"], name="Expected recoveries", marker_color="#2F6B5E")
        fig.update_layout(title="Monthly cash flows: contractual vs expected", barmode="overlay", xaxis_title="Month", yaxis_title=ui.amount_label("Cash flow"))
        ui.plot(fig, height=360)
    with c2:
        sweep = np.linspace(0, 0.5, 51)
        npvs = [fc.loan_npv(principal, rate, int(term), float(p_), fin["lgd"], disc, int(lag), fee_amt, serv)["npv_risk_adjusted"] for p_ in sweep]
        fig = go.Figure(go.Scatter(x=sweep, y=npvs, mode="lines", line_color="#1E4C8A", name="Risk-adjusted NPV"))
        fig.add_hline(y=0, line_color="#9AA8BC")
        fig.add_vline(x=apd, line_dash="dot", line_color=ui.INK, annotation_text="current PD")
        fig.update_layout(title="Risk-adjusted NPV as annual PD rises", xaxis=dict(title="Annual PD", tickformat=".0%"), yaxis_title=ui.amount_label("NPV"))
        ui.plot(fig, height=360)
    ui.download("Cash-flow schedule (CSV)", cf.to_csv(index=False), "loan_cash_flows.csv", "text/csv")
    ui.note("This is a simplified illustration, not an IFRS 9 lifetime ECL: the hazard is constant, there is no macroeconomic "
            "weighting, staging or prepayment model.")
