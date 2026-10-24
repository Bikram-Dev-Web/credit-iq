"""Shared test fixtures. Runtime folders are redirected to a temporary directory
before any CreditIQ module is imported, so tests never touch real records."""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

_TMP = Path(tempfile.mkdtemp(prefix="creditiq_tests_"))
os.environ["CREDITIQ_STORAGE_DIR"] = str(_TMP / "storage")
os.environ["CREDITIQ_MODELS_DIR"] = str(_TMP / "models")
os.environ["CREDITIQ_REPORTS_DIR"] = str(_TMP / "reports")

import pytest  # noqa: E402

from src import data_loader as dl  # noqa: E402
from src import data_validation as dv  # noqa: E402
from src import modeling as mdl  # noqa: E402


@pytest.fixture(scope="session")
def synthetic_df():
    return dl.generate_synthetic_credit_data(n_rows=4_000, seed=7)


@pytest.fixture(scope="session")
def mapping(synthetic_df):
    return dv.build_target_mapping(synthetic_df, "TARGET", [1], [0], "synthetic test target")


@pytest.fixture(scope="session")
def predictors(synthetic_df):
    roles = dv.suggest_feature_roles(synthetic_df, "TARGET", "APPLICATION_DATE")
    return roles.loc[roles["use"], "column"].tolist()


@pytest.fixture(scope="session")
def experiment(synthetic_df, mapping, predictors):
    cfg = mdl.TrainingConfig(
        sample_size=None, seed=7,
        params={"random_forest": {"n_estimators": 60, "max_depth": 8}, "gbm": {"learning_rate": 0.1}},
    )
    info = {"name": "test_synthetic.csv", "is_synthetic": True, "source": "synthetic"}
    return mdl.run_experiment(synthetic_df, info, mapping, predictors, cfg, date_col="APPLICATION_DATE",
                              exposure_col="AMT_CREDIT", sensitive_cols=["CODE_GENDER", "DAYS_BIRTH"],
                              excluded={"SK_ID_CURR": "identifier"})
