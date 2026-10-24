"""Shared fixtures. The core has no I/O, so unlike v1 no runtime folders need redirecting."""
from __future__ import annotations

import pytest

from creditiq_core.data import synthetic as synth
from creditiq_core.data import validation as dv
from creditiq_core.models import training as mdl


@pytest.fixture(scope="session")
def synthetic_df():
    return synth.generate_synthetic_credit_data(n_rows=4_000, seed=7)


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
