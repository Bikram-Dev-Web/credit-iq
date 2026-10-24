"""Headless UI smoke tests with Streamlit's AppTest: every page must render without
an exception, both before any data is loaded and after a full training run."""
from pathlib import Path

import pytest

AppTest = pytest.importorskip("streamlit.testing.v1").AppTest

APP = str(Path(__file__).resolve().parent.parent / "app.py")
PAGES = ["app_pages/1_Overview.py", "app_pages/2_Data_Center.py", "app_pages/3_EDA.py", "app_pages/4_Model_Lab.py",
         "app_pages/5_Borrower_Simulator.py", "app_pages/6_Portfolio_Intelligence.py", "app_pages/7_Financial_Scenarios.py",
         "app_pages/8_Explainability_Fairness.py", "app_pages/9_Model_Monitoring.py", "app_pages/10_Reports_Documentation.py"]


def _ok(at, where):
    errors = [str(e.value)[:400] for e in at.exception]
    assert not errors, f"{where}: {errors}"


def _button(at, label):
    for b in at.button:
        if b.label == label:
            return b
    raise AssertionError(f"button {label!r} not found")


def test_every_page_renders_empty():
    at = AppTest.from_file(APP, default_timeout=120)
    at.run()
    _ok(at, "overview")
    for p in PAGES[1:]:
        at.switch_page(p)
        at.run()
        _ok(at, p)


def test_full_workflow_renders_every_page():
    at = AppTest.from_file(APP, default_timeout=300)
    at.run()
    _button(at, "Load the synthetic demo dataset").click()
    at.run()
    _ok(at, "load demo")
    from src import data_validation as dv
    at.session_state["target"] = dv.build_target_mapping(at.session_state["df"], "TARGET", [1], [0], "synthetic")
    at.switch_page("app_pages/4_Model_Lab.py")
    at.run()
    _button(at, "Use the suggested predictor set").click()
    at.run()
    _ok(at, "suggested predictors")
    assert "POST_LOAN_COLLECTION_CALLS" not in at.session_state["predictors"]
    _button(at, "Train models").click()
    at.run()
    _ok(at, "training")
    assert at.session_state["experiment"] is not None
    for p in PAGES:
        at.switch_page(p)
        at.run()
        _ok(at, p)
    at.switch_page("app_pages/5_Borrower_Simulator.py")
    at.run()
    _button(at, "Assess applicant").click()
    at.run()
    _ok(at, "assess applicant")
    assert 0 <= at.session_state["last_assessment"]["pd"] <= 1
    at.switch_page("app_pages/9_Model_Monitoring.py")
    at.run()
    _button(at, "Run monitoring check").click()
    at.run()
    _ok(at, "monitoring")
