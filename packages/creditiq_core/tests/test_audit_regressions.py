"""Regression tests for defects found in the QA audit. Each test reproduces the original failure mode."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from creditiq_core import fairness as fair
from creditiq_core.data import csv as csvio
from creditiq_core.data import validation as dv
from creditiq_core.finance import cost_matrix as de
from creditiq_core.finance import economics as fc
from creditiq_core.models import training as mdl

ROOT = Path(__file__).resolve().parents[3]  # repository root


# ---------------------------------------------------------------- financial horizon conventions
def test_horizon_pd_and_inverse():
    assert fc.horizon_pd(0.10, 1) == pytest.approx(0.10)
    assert fc.horizon_pd(0.10, 2) == pytest.approx(0.19)
    assert fc.annual_pd_from_horizon(0.19, 2) == pytest.approx(0.10)
    with pytest.raises(ValueError):
        fc.horizon_pd(0.1, 0)


def test_two_year_contribution_uses_one_horizon_everywhere():
    # 1 loan of 100, annual PD 10% -> PD over 2 years = 19%; LGD 50% -> EL = 0.19 × 0.5 × 100 = 9.5
    a = fc.LendingAssumptions(n_loans=1, avg_exposure=100, pd=0.10, lgd=0.5, annual_rate=0.10, funding_rate=0.05,
                              opex_per_loan=1, horizon_years=2)
    r = fc.lending_contribution(a)
    assert r["pd_over_horizon"] == pytest.approx(0.19)
    assert r["expected_credit_loss"] == pytest.approx(9.5)
    assert r["expected_interest_income"] == pytest.approx(100 * 0.10 * 2 * 0.81)  # 16.2
    assert r["funding_cost"] == pytest.approx(10.0)
    assert r["operating_cost"] == pytest.approx(2.0)  # annual opex × 2 years
    assert r["expected_contribution"] == pytest.approx(16.2 - 10 - 9.5 - 2)


def test_break_even_pd_and_rate_at_two_years_zero_the_contribution():
    a = fc.LendingAssumptions(n_loans=10, avg_exposure=10_000, pd=0.05, lgd=0.45, annual_rate=0.16, funding_rate=0.07,
                              opex_per_loan=300, horizon_years=2)
    be_pd = fc.break_even_pd(a)
    assert be_pd is not None and 0 < be_pd < 1
    from dataclasses import replace
    assert fc.lending_contribution(replace(a, pd=be_pd))["expected_contribution"] == pytest.approx(0, abs=1e-6)
    be_rate = fc.break_even_rate(a)
    assert fc.lending_contribution(replace(a, annual_rate=be_rate))["expected_contribution"] == pytest.approx(0, abs=1e-6)


def test_stress_base_matches_contribution_at_horizon():
    p = np.full(50, 0.1)
    e = np.full(50, 100.0)
    s = fc.stress_test(p, 0.5, e, [{"scenario": "Base", "pd_multiplier": 1, "lgd_shift": 0, "ead_multiplier": 1}],
                       0.10, 0.05, 1.0, horizon_years=2).iloc[0]
    a = fc.LendingAssumptions(n_loans=50, avg_exposure=100, pd=0.1, lgd=0.5, annual_rate=0.10, funding_rate=0.05,
                              opex_per_loan=1.0, horizon_years=2)
    r = fc.lending_contribution(a)
    assert s["avg_pd_over_horizon"] == pytest.approx(0.19)
    assert s["expected_loss"] == pytest.approx(r["expected_credit_loss"])
    assert s["expected_contribution"] == pytest.approx(r["expected_contribution"])


def test_policy_expected_contribution_matches_lending_contribution():
    a = fc.LendingAssumptions(n_loans=1, avg_exposure=1_000, pd=0.2, lgd=0.4, annual_rate=0.2, funding_rate=0.05,
                              opex_per_loan=10, horizon_years=3)
    per_loan = de.expected_contribution_per_loan([0.2], [1_000], 0.4, 0.2, 0.05, 10, 3)[0]
    assert per_loan == pytest.approx(fc.lending_contribution(a)["expected_contribution"])


def test_realized_contribution_scales_opex_with_horizon():
    y = np.array([0, 1])
    approved = np.array([True, True])
    ead = np.array([100.0, 100.0])
    v = de.realized_contribution(y, approved, ead, lgd=0.5, annual_rate=0.1, funding_rate=0.05, opex_per_loan=1, horizon_years=2)
    good = 100 * (0.1 - 0.05) * 2
    bad = 100 * (0 - 0.05 * 2 - 0.5)
    assert v == pytest.approx(good + bad - 1 * 2 * 2)


def test_npv_default_is_costless_with_zero_lgd_no_lag_and_loan_rate_discounting():
    r = fc.loan_npv(100_000, 0.12, 24, 0.30, lgd=0.0, discount_rate=0.12, recovery_lag_months=0)
    assert r["npv_contractual"] == pytest.approx(0, abs=1e-6)
    assert r["npv_risk_adjusted"] == pytest.approx(r["npv_contractual"], abs=1e-6)
    r2 = fc.loan_npv(100_000, 0.12, 24, 0.30, lgd=0.45, discount_rate=0.12, recovery_lag_months=6)
    assert r2["npv_risk_adjusted"] < r2["npv_contractual"]


# ---------------------------------------------------------------- data validation
def test_target_mapping_matches_float_and_string_labels():
    s = pd.Series([1.0, 0.0, 1.0, np.nan, 0.0])
    m = dv.apply_target_mapping(s, [1], [0])  # integers chosen in the UI, floats in the file
    assert m.tolist()[:3] == [1.0, 0.0, 1.0] and np.isnan(m.iloc[3])
    s2 = pd.Series(["1", "0", " 1 ", "1.0"])
    assert dv.apply_target_mapping(s2, [1], [0]).tolist() == [1.0, 0.0, 1.0, 1.0]
    assert dv.label_key(np.int64(1)) == dv.label_key("1") == dv.label_key(1.0) == "1"
    assert dv.label_key("Charged Off") == "Charged Off"


def test_build_target_mapping_negative_values_with_float_column():
    df = pd.DataFrame({"y": [1.0, 0.0] * 50})
    m = dv.build_target_mapping(df, "y", [1], None, "test")
    assert m.n_positive == 50 and m.n_negative == 50 and m.n_unmapped == 0


def test_sequential_integer_id_is_flagged_but_amounts_are_not():
    rng = np.random.default_rng(0)
    n = 500
    df = pd.DataFrame({"row_number": np.arange(10_000, 10_000 + n), "amount": rng.integers(1_000, 5_000_000, n),
                       "OWN_CAR_AGE": rng.integers(0, 20, n), "applicant_age": rng.integers(20, 70, n),
                       "TARGET": rng.integers(0, 2, n)})
    roles = dv.suggest_feature_roles(df, "TARGET").set_index("column")
    assert roles.loc["row_number", "role"] == "identifier"
    assert roles.loc["amount", "role"] != "identifier"
    assert roles.loc["OWN_CAR_AGE", "role"] != "sensitive"  # 'car_age' is not a person's age
    assert roles.loc["applicant_age", "role"] == "sensitive"


# ---------------------------------------------------------------- CSV ingestion
def test_semicolon_and_latin1_csv_are_parsed(tmp_path):
    text = "id;income;city;TARGET\n1;1000,5;Zürich;0\n2;2000;Köln;1\n3;3000;Paris;0\n"
    df, info = csvio.read_csv_file(text.encode("latin-1"), name="semi.csv", source_kind="upload")
    assert list(df.columns) == ["id", "income", "city", "TARGET"]
    assert df.shape == (3, 4) and df["city"].iloc[0] == "Zürich"
    assert any(";" in n for n in info.notes)
    tab = "a\tb\n1\t2\n3\t4\n"
    df2, _ = csvio.read_csv_file(tab.encode(), name="tab.csv", source_kind="upload")
    assert list(df2.columns) == ["a", "b"]


def test_bundled_sample_is_flagged_synthetic_when_uploaded():
    data = (ROOT / "data" / "sample_credit_data.csv").read_bytes()
    _, info = csvio.read_csv_file(data, name="renamed_copy.csv", source_kind="upload")
    assert info.is_synthetic


# ---------------------------------------------------------------- modelling: leakage guard and calibration
def _quick_cfg(**kw):
    return mdl.TrainingConfig(model_keys=["logreg"], sample_size=None, seed=3, **kw)


def test_leakage_guard_screens_training_rows_only(synthetic_df, mapping):
    preds = ["EXT_SOURCE_2", "AMT_CREDIT", "POST_LOAN_COLLECTION_CALLS"]
    info = {"name": "t", "is_synthetic": True}
    guarded = mdl.run_experiment(synthetic_df, info, mapping, preds, _quick_cfg())
    assert "POST_LOAN_COLLECTION_CALLS" not in guarded.predictors
    assert guarded.excluded["POST_LOAN_COLLECTION_CALLS"].startswith("leakage (training-split")
    assert any("leakage guard" in w for w in guarded.warnings)
    unguarded = mdl.run_experiment(synthetic_df, info, mapping, preds, _quick_cfg(leakage_guard=False))
    assert "POST_LOAN_COLLECTION_CALLS" in unguarded.predictors
    # the leaky column inflates held-out AUC — exactly what the guard prevents
    assert unguarded.metrics["logreg"]["test"]["roc_auc"] > guarded.metrics["logreg"]["test"]["roc_auc"]


def test_leakage_guard_errors_when_nothing_is_left(synthetic_df, mapping):
    with pytest.raises(ValueError):
        mdl.run_experiment(synthetic_df, {"name": "t"}, mapping, ["POST_LOAN_COLLECTION_CALLS"], _quick_cfg())


def test_uncalibrated_model_is_reported_as_such(synthetic_df, mapping):
    r = mdl.run_experiment(synthetic_df, {"name": "t"}, mapping, ["EXT_SOURCE_2", "AMT_CREDIT"], _quick_cfg(calibration="none"))
    assert r.models["logreg"].is_calibrated is False
    assert r.metadata()["calibrated"] == {"logreg": False}
    assert any("not probabilities" in w for w in r.warnings)
    r2 = mdl.run_experiment(synthetic_df, {"name": "t"}, mapping, ["EXT_SOURCE_2", "AMT_CREDIT"], _quick_cfg())
    assert r2.models["logreg"].is_calibrated is True


def test_threshold_and_calibration_never_see_test_rows(experiment):
    tr, va, te = (set(experiment.idx(s)) for s in ("train", "validation", "test"))
    assert not (tr & va) and not (tr & te) and not (va & te)
    # the stored threshold is reproducible from validation data alone
    from creditiq_core import evaluation as ev
    key = experiment.selected
    t = ev.select_threshold(experiment.y_split("validation"), experiment.pd_split("validation", key),
                            experiment.config.threshold_rule, manual=experiment.config.manual_threshold)
    assert t == pytest.approx(experiment.thresholds[key])


# ---------------------------------------------------------------- fairness naming
def test_fairness_reports_within_threshold_not_approval():
    y = np.array([0, 1] * 60)
    p = np.linspace(0.01, 0.99, 120)
    g = pd.Series(["a"] * 60 + ["b"] * 60)
    gm = fair.group_metrics(y, p, 0.5, g, min_group=10)
    assert "within_threshold_rate" in gm.columns and "approval_rate" not in gm.columns
    assert (gm["within_threshold_rate"] + gm["flag_rate"]).round(9).eq(1).all()
    assert fair.cramers_v(pd.Series(list("aabb") * 10), pd.Series(list("xyxy") * 10)) is not None


# ---------------------------------------------------------------- currency formatting
# ---------------------------------------------------------------- invalid inputs
def test_header_only_and_binary_csv_are_rejected_clearly():
    with pytest.raises(ValueError, match="no data rows"):
        csvio.read_csv_file(b"a,b,TARGET\n", name="h.csv", source_kind="upload")
    with pytest.raises(ValueError, match="binary"):
        csvio.read_csv_file(bytes(range(256)) * 50, name="g.csv", source_kind="upload")


def test_all_missing_predictor_is_dropped_with_a_warning(synthetic_df, mapping):
    df = synthetic_df.assign(ALL_MISSING=np.nan)
    r = mdl.run_experiment(df, {"name": "t"}, mapping, ["EXT_SOURCE_2", "AMT_CREDIT", "ALL_MISSING"], _quick_cfg())
    assert "ALL_MISSING" not in r.predictors and r.excluded["ALL_MISSING"] == "empty in training split"
    assert any("no observed values" in w for w in r.warnings)
