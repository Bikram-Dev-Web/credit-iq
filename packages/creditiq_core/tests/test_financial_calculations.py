import math
from dataclasses import replace

import numpy as np
import pytest

from creditiq_core.finance import economics as fc


def test_expected_loss_scalar():
    assert fc.expected_loss(0.05, 0.45, 100_000) == pytest.approx(2_250.0)


def test_expected_loss_vectorised_and_portfolio_sum():
    pd_ = np.array([0.01, 0.10, 0.50])
    lgd = np.array([0.40, 0.45, 0.60])
    ead = np.array([1_000.0, 2_000.0, 3_000.0])
    el = fc.expected_loss(pd_, lgd, ead)
    assert el == pytest.approx([4.0, 90.0, 900.0])
    port = fc.portfolio_expected_loss(pd_, lgd, ead)
    assert port["total_el"] == pytest.approx(994.0)  # sum of loan-level ELs, not avg PD × avg LGD × total EAD
    assert port["total_ead"] == pytest.approx(6_000.0)
    assert port["el_rate"] == pytest.approx(994.0 / 6_000.0)


@pytest.mark.parametrize("pd_, lgd, ead", [(1.2, 0.4, 100), (-0.1, 0.4, 100), (0.1, 1.5, 100), (0.1, -0.2, 100),
                                           (0.1, 0.4, -5), (float("nan"), 0.4, 100), (0.1, 0.4, float("inf"))])
def test_expected_loss_rejects_invalid_inputs(pd_, lgd, ead):
    with pytest.raises(ValueError):
        fc.expected_loss(pd_, lgd, ead)


def test_expected_loss_bounds_edges_are_valid():
    assert fc.expected_loss(0.0, 0.45, 100) == 0.0
    assert fc.expected_loss(1.0, 1.0, 100) == 100.0


def test_contribution_components_and_no_double_counting():
    a = fc.LendingAssumptions(n_loans=100, avg_exposure=10_000, pd=0.1, lgd=0.5, annual_rate=0.2,
                              funding_rate=0.05, opex_per_loan=100)
    r = fc.lending_contribution(a)
    exposure = 1_000_000
    assert r["total_exposure"] == exposure
    assert r["expected_interest_income"] == pytest.approx(exposure * 0.2 * 0.9)
    assert r["funding_cost"] == pytest.approx(exposure * 0.05)
    assert r["expected_credit_loss"] == pytest.approx(100 * 0.1 * 0.5 * 10_000)
    assert r["operating_cost"] == pytest.approx(10_000)
    expected = r["expected_interest_income"] - r["funding_cost"] - r["expected_credit_loss"] - r["operating_cost"]
    assert r["expected_contribution"] == pytest.approx(expected)


def test_break_even_pd_and_rate_give_zero_contribution():
    a = fc.LendingAssumptions(n_loans=10, avg_exposure=50_000, pd=0.06, lgd=0.45, annual_rate=0.18,
                              funding_rate=0.07, opex_per_loan=500)
    p_star = fc.break_even_pd(a)
    assert 0 < p_star < 1
    assert fc.lending_contribution(replace(a, pd=p_star))["expected_contribution"] == pytest.approx(0, abs=1e-6)
    r_star = fc.break_even_rate(a)
    assert fc.lending_contribution(replace(a, annual_rate=r_star))["expected_contribution"] == pytest.approx(0, abs=1e-6)


def test_contribution_validation_errors():
    with pytest.raises(ValueError):
        fc.lending_contribution(fc.LendingAssumptions(pd=1.5))
    with pytest.raises(ValueError):
        fc.lending_contribution(fc.LendingAssumptions(n_loans=-1))
    with pytest.raises(ValueError):
        fc.lending_contribution(fc.LendingAssumptions(annual_rate=-0.1))


def test_contribution_falls_as_pd_rises():
    base = fc.LendingAssumptions()
    values = [fc.lending_contribution(replace(base, pd=p))["expected_contribution"] for p in (0.0, 0.05, 0.1, 0.2)]
    assert all(x > y for x, y in zip(values, values[1:]))


def test_stress_test_scenarios_and_caps():
    pd_ = np.array([0.05, 0.6])
    ead = np.array([100.0, 100.0])
    scenarios = [{"scenario": "Base", "pd_multiplier": 1.0, "lgd_shift": 0.0, "ead_multiplier": 1.0},
                 {"scenario": "Adverse", "pd_multiplier": 2.0, "lgd_shift": 0.7, "ead_multiplier": 1.0}]
    out = fc.stress_test(pd_, 0.45, ead, scenarios, 0.15, 0.05, 10)
    base, adverse = out.iloc[0], out.iloc[1]
    assert base["expected_loss"] == pytest.approx(0.05 * 0.45 * 100 + 0.6 * 0.45 * 100)
    # PD (1.2 → 1.0) and LGD (1.15 → 1.0) are capped at 100%
    assert adverse["expected_loss"] == pytest.approx(0.10 * 1.0 * 100 + 1.0 * 1.0 * 100)
    assert adverse["expected_contribution"] < base["expected_contribution"]
    with pytest.raises(ValueError):
        fc.stress_test(pd_, 0.45, ead, [{"scenario": "x", "pd_multiplier": -1, "lgd_shift": 0, "ead_multiplier": 1}], 0.1, 0.05, 0)


def test_scenario_comparison_orders_default_scenarios():
    out = fc.stress_test(np.full(50, 0.08), 0.45, np.full(50, 1_000.0), fc.DEFAULT_SCENARIOS, 0.16, 0.07, 10)
    el = dict(zip(out["scenario"], out["expected_loss"]))
    assert el["Improved"] < el["Base"] < el["Adverse"] < el["Severe stress"]


def test_annuity_payment_known_value():
    # 100,000 over 12 months at 12% a year → 8,884.88 per month (standard annuity formula)
    assert fc.annuity_payment(100_000, 0.12, 12) == pytest.approx(8884.88, abs=0.01)
    sched = fc.amortization_schedule(100_000, 0.12, 12)
    assert sched["closing_balance"].iloc[-1] == pytest.approx(0, abs=1e-6)
    assert sched["principal"].sum() == pytest.approx(100_000, rel=1e-9)


def test_loan_npv_without_default_equals_contractual():
    r = fc.loan_npv(100_000, 0.12, 24, 0.0, 0.45, 0.06)
    assert r["npv_risk_adjusted"] == pytest.approx(r["npv_contractual"])
    assert r["pv_credit_cost"] == pytest.approx(0, abs=1e-6)
    assert r["contractual_annual_yield"] == pytest.approx((1 + 0.12 / 12) ** 12 - 1, rel=1e-4)


def test_loan_npv_decreases_with_pd_and_lifetime_pd():
    npvs = [fc.loan_npv(100_000, 0.15, 36, p, 0.45, 0.07)["npv_risk_adjusted"] for p in (0.0, 0.05, 0.2)]
    assert npvs[0] > npvs[1] > npvs[2]
    r = fc.loan_npv(100_000, 0.15, 12, 0.10, 0.45, 0.07)
    assert r["lifetime_pd"] == pytest.approx(0.10, rel=1e-6)  # 12-month term → lifetime PD equals annual PD
    assert math.isclose(fc.monthly_hazard(0.10), 1 - 0.9 ** (1 / 12))


def test_sensitivity_analysis_shape():
    out = fc.sensitivity_analysis(fc.LendingAssumptions(), 0.2)
    assert set(out["driver"]) == {"PD", "LGD", "Average exposure", "Lending rate", "Funding rate", "Operating cost per loan"}
    pd_row = out[out["driver"] == "PD"].iloc[0]
    assert pd_row["change_if_driver_up"] < 0 < pd_row["change_if_driver_down"]
