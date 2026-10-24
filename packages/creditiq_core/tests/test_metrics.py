import numpy as np
import pytest

from creditiq_core import evaluation as ev

Y = np.array([0, 0, 0, 0, 1, 1, 0, 1])
P = np.array([0.1, 0.2, 0.3, 0.6, 0.7, 0.4, 0.05, 0.9])


def test_confusion_and_threshold_metrics():
    m = ev.classification_metrics(Y, P, 0.5)
    # flagged (p ≥ 0.5): idx 3 (y=0), 4 (y=1), 7 (y=1)
    assert (m["tp"], m["fp"], m["tn"], m["fn"]) == (2, 1, 4, 1)
    assert m["precision"] == pytest.approx(2 / 3)
    assert m["recall"] == pytest.approx(2 / 3)
    assert m["specificity"] == pytest.approx(4 / 5)
    assert m["balanced_accuracy"] == pytest.approx((2 / 3 + 4 / 5) / 2)
    assert m["prevalence"] == pytest.approx(3 / 8)
    assert m["brier"] == pytest.approx(np.mean((P - Y) ** 2))
    assert 0.5 < m["roc_auc"] <= 1


def test_undefined_metrics_are_none():
    m = ev.classification_metrics([0, 0, 0], [0.1, 0.2, 0.3], 0.5)
    assert m["roc_auc"] is None and m["pr_auc"] is None and m["ks"] is None
    assert m["precision"] is None  # nobody flagged
    assert m["recall"] is None  # no defaulters


def test_invalid_probabilities_rejected():
    with pytest.raises(ValueError):
        ev.classification_metrics([0, 1], [0.2, 1.4])
    with pytest.raises(ValueError):
        ev.classification_metrics([], [])


def test_threshold_sweep_matches_direct_metrics():
    sweep = ev.threshold_sweep(Y, P, [0.25, 0.5, 0.75]).set_index("threshold")
    for t in (0.25, 0.5, 0.75):
        m = ev.classification_metrics(Y, P, t)
        row = sweep.loc[t]
        assert (row["tp"], row["fp"], row["tn"], row["fn"]) == (m["tp"], m["fp"], m["tn"], m["fn"])


def test_threshold_decisions_monotonic():
    sweep = ev.threshold_sweep(Y, P)
    assert (np.diff(sweep["flagged"]) <= 0).all()


def test_ks_and_lift():
    ks, _ = ev.ks_statistic(Y, P)
    assert 0 < ks <= 1
    assert ev.lift_at(Y, P, 0.25) == pytest.approx(1.0 / (3 / 8))  # top 2 records are both defaulters


def test_calibration_table_and_wilson():
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 0.4, 5000)
    y = (rng.uniform(0, 1, 5000) < p).astype(int)  # perfectly calibrated by construction
    tab = ev.calibration_table(y, p, 10)
    assert tab["n"].sum() == 5000
    assert ev.expected_calibration_error(tab) < 0.02
    lo, hi = ev.wilson_interval(10, 100)
    assert lo < 0.1 < hi


def test_select_threshold_rules():
    rng = np.random.default_rng(1)
    p = rng.uniform(0, 1, 2000)
    y = (rng.uniform(0, 1, 2000) < p).astype(int)
    assert ev.select_threshold(y, p, "manual", 0.33) == 0.33
    t_prev = ev.select_threshold(y, p, "prevalence")
    assert (p >= t_prev).mean() == pytest.approx(y.mean(), abs=0.01)
    assert 0 < ev.select_threshold(y, p, "max_f1") < 1
    assert 0 < ev.select_threshold(y, p, "youden") < 1


def test_decile_table():
    rng = np.random.default_rng(2)
    p = rng.uniform(0, 1, 1000)
    y = (rng.uniform(0, 1, 1000) < p).astype(int)
    d = ev.decile_table(y, p)
    assert len(d) == 10 and d["n"].sum() == 1000
    assert d["default_rate"].iloc[0] > d["default_rate"].iloc[-1]
    assert d["cum_capture"].iloc[-1] == pytest.approx(1.0)


def test_metrics_by_period_handles_missing_dates():
    import pandas as pd
    dates = pd.Series(["2024-01-05", "2024-02-01", None, "2024-04-01", "2024-05-01", "2024-07-01", "bad", "2024-08-01"])
    out = ev.metrics_by_period(Y, P, dates, 0.5, "Q")
    assert out["n"].sum() == 6
