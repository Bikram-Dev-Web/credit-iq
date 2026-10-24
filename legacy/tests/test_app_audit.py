"""Headless UI regression tests for audit findings: small datasets, the threshold explorer, uncalibrated
models and invalidation of results when the active model or threshold changes."""
from pathlib import Path

import pytest

AppTest = pytest.importorskip("streamlit.testing.v1").AppTest

from src import data_loader as dl  # noqa: E402
from src import data_validation as dv  # noqa: E402

APP = str(Path(__file__).resolve().parent.parent / "app.py")


def _ok(at, where):
    errors = [str(e.value)[:400] for e in at.exception]
    assert not errors, f"{where}: {errors}"


def _button(at, label):
    for b in at.button:
        if b.label == label:
            return b
    raise AssertionError(f"button {label!r} not found")


def _has_button(at, label):
    return any(b.label == label for b in at.button)


def _widget(group, label):
    for w in group:
        if w.label == label:
            return w
    raise AssertionError(f"widget {label!r} not found")


def _trained_app(n_rows, models=("Logistic regression",), calibration=None):
    at = AppTest.from_file(APP, default_timeout=300)
    at.run()
    df, info = dl.load_synthetic_dataset(n_rows, 11)
    at.session_state["df"], at.session_state["ds_info"] = df, info
    at.session_state["target"] = dv.build_target_mapping(df, "TARGET", [1], [0], "synthetic")
    at.switch_page("app_pages/4_Model_Lab.py")
    at.run()
    _button(at, "Use the suggested predictor set").click()
    at.run()
    _ok(at, "suggested predictors")
    ms = _widget(at.multiselect, "Models")
    ms.set_value([o for o in ms.options if any(o.lower().startswith(m.lower()) for m in models)])
    if calibration:
        sb = _widget(at.selectbox, "Probability calibration")
        sb.set_value(next(o for o in sb.options if o.lower().startswith(calibration)))
    _button(at, "Train models").click()
    at.run()
    _ok(at, "training")
    assert at.session_state["experiment"] is not None, [e.value for e in at.error]
    return at


def test_small_dataset_trains_in_model_lab():
    # fewer labelled rows than the 1,000-row floor of the sample-size input used to crash the page
    at = _trained_app(600)
    exp = at.session_state["experiment"]
    assert sum(exp.split.sizes().values()) == 600


def test_threshold_explorer_defaults_to_validation_and_test_view_is_read_only():
    at = _trained_app(3000)
    radio = _widget(at.radio, "Evaluate on")
    assert radio.value.startswith("Validation")
    assert _has_button(at, "Use this threshold across the app")
    radio.set_value("Test (held-out, read-only)")
    at.run()
    _ok(at, "test view")
    assert not _has_button(at, "Use this threshold across the app")


def test_threshold_override_invalidates_results_and_reaches_policy_page():
    at = _trained_app(3000)
    exp = at.session_state["experiment"]
    key = at.session_state["active_model_key"]
    at.switch_page("app_pages/7_Financial_Scenarios.py")
    at.run()
    _ok(at, "financial")
    assert at.session_state["policy_table"] is not None
    at.switch_page("app_pages/4_Model_Lab.py")
    at.run()
    _widget(at.slider, next(s.label for s in at.slider if s.label.startswith("Decision threshold on"))).set_value(37.0)
    at.run()
    _button(at, "Use this threshold across the app").click()
    at.run()
    _ok(at, "apply threshold")
    assert at.session_state["threshold_override"][(exp.experiment_id, key)] == pytest.approx(0.37)
    assert at.session_state["policy_table"] is None  # stale results dropped
    at.switch_page("app_pages/7_Financial_Scenarios.py")
    at.run()
    _ok(at, "financial after override")
    assert any("37.0%" in p for p in at.session_state["policy_table"]["policy"])


def test_switching_active_model_clears_downstream_results():
    at = _trained_app(3000, models=("Logistic regression", "Random forest"))
    at.switch_page("app_pages/7_Financial_Scenarios.py")
    at.run()
    assert at.session_state["policy_table"] is not None and at.session_state["scenario_table"] is not None
    at.switch_page("app_pages/4_Model_Lab.py")
    at.run()
    radio = next(r for r in at.radio if r.label.startswith("Active model"))
    current = at.session_state["active_model_key"]
    other = next(o for o in radio.options if not o.startswith({"logreg": "Logistic", "random_forest": "Random"}[current]))
    radio.set_value(other)
    at.run()
    _ok(at, "switch model")
    assert at.session_state["active_model_key"] != current
    assert at.session_state["policy_table"] is None and at.session_state["scenario_table"] is None


def test_uncalibrated_model_is_labelled_as_a_score_in_the_simulator():
    at = _trained_app(3000, calibration="no calibration")
    assert at.session_state["experiment"].selected_model.is_calibrated is False
    at.switch_page("app_pages/5_Borrower_Simulator.py")
    at.run()
    _button(at, "Assess applicant").click()
    at.run()
    _ok(at, "assess")
    html = " ".join(m.value for m in at.markdown)
    assert "Uncalibrated model score" in html and "not calibrated" in html


def _tile_values(at):
    import re
    html = " ".join(m.value for m in at.markdown)
    return dict(re.findall(r"ciq-tile-label'[^>]*>([^<]+)</div><div class='ciq-tile-value'>([^<]+)<", html))


def test_portfolio_filters_change_the_numbers():
    at = _trained_app(3000)
    exp = at.session_state["experiment"]
    at.switch_page("app_pages/6_Portfolio_Intelligence.py")
    at.run()
    _ok(at, "portfolio")
    n_all = len(exp.frame_split("test"))
    assert _tile_values(at)["Loans evaluated"] == f"{n_all:,}"
    level_filter = _widget(at.multiselect, "NAME_CONTRACT_TYPE")
    keep = level_filter.options[0]
    level_filter.set_value([keep])
    at.run()
    _ok(at, "portfolio filtered")
    expected = int((exp.frame_split("test")["NAME_CONTRACT_TYPE"].astype(str) == keep).sum())
    assert 0 < expected < n_all
    assert _tile_values(at)["Loans evaluated"] == f"{expected:,}"


def test_financial_horizon_input_drives_the_calculation():
    at = _trained_app(3000)
    at.switch_page("app_pages/7_Financial_Scenarios.py")
    at.run()
    one_year = at.session_state["contribution_result"]
    _widget(at.number_input, "Horizon (years)").set_value(2.0)
    at.run()
    _ok(at, "two-year horizon")
    r = at.session_state["contribution_result"]
    p = r["assumptions"]["pd"]
    assert r["assumptions"]["horizon_years"] == 2.0
    assert r["pd_over_horizon"] == pytest.approx(1 - (1 - p) ** 2)
    assert r["operating_cost"] == pytest.approx(2 * one_year["operating_cost"])
    assert r["funding_cost"] == pytest.approx(2 * one_year["funding_cost"])
    assert at.session_state["scenario_table"]["avg_pd_over_horizon"].iloc[1] > at.session_state["scenario_table"]["avg_pd"].iloc[1]


def test_privacy_controls_delete_records_and_clear_the_session():
    at = _trained_app(1500)
    at.switch_page("app_pages/10_Reports_Documentation.py")
    at.run()
    _ok(at, "reports")
    _widget(at.checkbox, "Experiment log").check()
    _widget(at.checkbox, "I understand this permanently deletes the selected local records.").check()
    at.run()
    _button(at, "Delete selected").click()
    at.run()
    _ok(at, "delete records")
    assert any("Deleted:" in s.value and "experiments" in s.value for s in at.success)
    _button(at, "Clear everything in this browser session (data, target, models)").click()
    at.run()
    _ok(at, "clear session")
    assert at.session_state["df"] is None and at.session_state["experiment"] is None
