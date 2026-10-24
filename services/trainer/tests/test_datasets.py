"""Dataset tooling on schema-compatible MOCK files (the real Kaggle/UCI files are never committed)."""
import pandas as pd
import pytest

from creditiq_core.data import validation as dv
from creditiq_trainer.datasets import german as gc
from creditiq_trainer.datasets import home_credit as hc


def test_home_credit_enrichment_on_mock_files(tmp_path):
    app = pd.DataFrame({"SK_ID_CURR": [1, 2, 3], "TARGET": [0, 1, 0], "AMT_CREDIT": [100.0, 200.0, 300.0]})
    bureau = pd.DataFrame({"SK_ID_CURR": [1, 1, 2], "SK_ID_BUREAU": [10, 11, 12], "CREDIT_ACTIVE": ["Active", "Closed", "Active"],
                           "DAYS_CREDIT": [-100, -300, -50], "CREDIT_DAY_OVERDUE": [0, 5, 0],
                           "AMT_CREDIT_SUM": [1000.0, 500.0, 0.0], "AMT_CREDIT_SUM_DEBT": [400.0, 0.0, 0.0],
                           "AMT_CREDIT_SUM_OVERDUE": [0.0, 10.0, 0.0], "CNT_CREDIT_PROLONG": [0, 1, 0]})
    prev = pd.DataFrame({"SK_ID_CURR": [1, 2, 2], "NAME_CONTRACT_STATUS": ["Approved", "Refused", "Approved"],
                         "AMT_APPLICATION": [50.0, 60.0, 70.0], "AMT_CREDIT": [50.0, 0.0, 70.0], "DAYS_DECISION": [-10, -20, -5]})
    for name, df in (("app.csv", app), ("bureau.csv", bureau), ("prev.csv", prev)):
        df.to_csv(tmp_path / name, index=False)
    out = hc.build(tmp_path / "app.csv", tmp_path / "bureau.csv", tmp_path / "prev.csv", tmp_path / "out.csv").set_index("SK_ID_CURR")
    assert out.loc[1, "BUREAU_LOAN_COUNT"] == 2 and out.loc[1, "BUREAU_ACTIVE_COUNT"] == 1
    assert out.loc[1, "BUREAU_DAYS_CREDIT_MEAN"] == -200 and out.loc[1, "BUREAU_MAX_DAYS_OVERDUE"] == 5
    assert out.loc[1, "BUREAU_DEBT_TO_CREDIT"] == pytest.approx(400 / 1500)
    assert out.loc[2, "PREV_REFUSED_SHARE"] == 0.5 and out.loc[2, "PREV_DAYS_DECISION_LATEST"] == -5
    assert pd.isna(out.loc[3, "BUREAU_LOAN_COUNT"])  # no history → missing, not zero
    bad = tmp_path / "bad.csv"
    pd.DataFrame({"SK_ID_CURR": [1]}).to_csv(bad, index=False)
    with pytest.raises(SystemExit):
        hc.aggregate_bureau(bad)


def test_german_credit_conversion_on_mock_lines(tmp_path):
    # Two lines in the documented german.data layout (20 attributes + class), written for this test.
    raw = ("A11 6 A34 A43 1169 A65 A75 4 A93 A101 4 A121 67 A143 A152 2 A173 1 A192 A201 1\n"
           "A12 48 A32 A43 5951 A61 A73 2 A92 A101 2 A121 22 A143 A152 1 A173 1 A191 A201 2\n")
    (tmp_path / "german.data").write_text(raw)
    df = gc.convert(tmp_path / "german.data", tmp_path / "german.csv")
    assert list(df.columns) == gc.COLUMNS and df["credit_risk_class"].tolist() == [1, 2]
    roles = dv.suggest_feature_roles(df, "credit_risk_class").set_index("column")
    assert roles.loc["personal_status_sex", "role"] == "sensitive" and roles.loc["age_years", "role"] == "sensitive"
