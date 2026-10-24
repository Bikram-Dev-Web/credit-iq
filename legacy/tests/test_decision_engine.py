import numpy as np
import pytest

from src import decision_engine as de
from src import financial_calculations as fc


def test_evaluate_policy_costs():
    y = np.array([1, 0, 1, 0, 0])
    flagged = np.array([True, True, False, False, False])
    r = de.evaluate_policy(y, flagged, c_fn=np.full(5, 1000.0), c_fp=np.full(5, 100.0), review_cost=10)
    assert (r["tp"], r["fp"], r["tn"], r["fn"]) == (1, 1, 2, 1)
    assert r["missed_default_cost"] == 1000 and r["rejected_good_cost"] == 100 and r["review_cost_total"] == 20
    assert r["total_cost"] == 1120
    assert r["approval_rate"] == pytest.approx(3 / 5)


def test_false_positive_and_negative_costs_differ():
    y = np.array([1, 0])
    c_fn, c_fp = de.cost_vectors(de.CostAssumptions(mode="fixed", missed_default_cost=500, rejected_good_cost=50), None, 2)
    miss = de.evaluate_policy(y, [False, False], c_fn, c_fp, 0)["total_cost"]
    reject = de.evaluate_policy(y, [True, True], c_fn, c_fp, 0)["total_cost"]
    assert miss == 500 and reject == 50


def test_per_loan_costs_scale_with_ead():
    c_fn, c_fp = de.cost_vectors(de.CostAssumptions(mode="per_loan", lgd=0.4, lost_margin_rate=0.05), np.array([100.0, 1000.0]), 2)
    assert c_fn.tolist() == [40.0, 400.0] and c_fp.tolist() == [5.0, 50.0]
    with pytest.raises(ValueError):
        de.cost_vectors(de.CostAssumptions(mode="per_loan"), None, 2)
    with pytest.raises(ValueError):
        de.CostAssumptions(review_cost=-1).validate()


def test_cost_optimal_threshold_minimises_cost():
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, 3000)
    y = (rng.uniform(0, 1, 3000) < p).astype(int)
    c_fn, c_fp = np.full(3000, 10.0), np.full(3000, 1.0)
    t = de.cost_optimal_threshold(y, p, c_fn, c_fp, 0.0)
    curve = de.cost_curve(y, p, c_fn, c_fp, 0.0)
    best_cost = de.evaluate_policy(y, p >= t, c_fn, c_fp, 0.0)["total_cost"]
    assert best_cost <= curve["total_cost"].min() + 1e-9
    assert 0.0 < t < 0.3  # costly misses push the threshold down (theory: 1/(1+10) ≈ 0.09)


def test_expected_contribution_rule_matches_calculator():
    ec = de.expected_contribution_per_loan(np.array([0.05]), np.array([10_000.0]), 0.45, 0.16, 0.07, 500)
    a = fc.LendingAssumptions(n_loans=1, avg_exposure=10_000, pd=0.05, lgd=0.45, annual_rate=0.16, funding_rate=0.07, opex_per_loan=500)
    assert ec[0] == pytest.approx(fc.lending_contribution(a)["expected_contribution"])


def test_realized_contribution():
    y = np.array([0, 1, 0])
    approved = np.array([True, True, False])
    ead = np.array([1000.0, 1000.0, 1000.0])
    got = de.realized_contribution(y, approved, ead, 0.5, 0.2, 0.05, 10)
    assert got == pytest.approx(1000 * 0.15 + (-1000 * 0.05 - 500) - 20)


def test_compare_policies(experiment):
    exp = experiment
    ead_v = exp.frame_split("validation")["AMT_CREDIT"].to_numpy(float)
    ead_t = exp.frame_split("test")["AMT_CREDIT"].to_numpy(float)
    table, extra = de.compare_policies(exp.y_split("validation"), exp.pd_split("validation"), ead_v,
                                       exp.y_split("test"), exp.pd_split("test"), ead_t, 0.15,
                                       de.CostAssumptions(), 0.16, 0.07, 500)
    assert len(table) == 4
    baseline = table.iloc[0]
    assert baseline["flagged"] == 0 and baseline["fn"] == exp.y_split("test").sum()
    assert (table[["tp", "fp", "tn", "fn"]].sum(axis=1) == len(exp.y_split("test"))).all()
    assert 0 <= extra["cost_optimal_threshold"] <= 1
