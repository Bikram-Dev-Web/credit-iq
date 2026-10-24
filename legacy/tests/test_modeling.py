import importlib.util

import numpy as np
import pandas as pd
import pytest

from src import data_validation as dv
from src import explainability as xai
from src import modeling as mdl
from src import persistence as ps
from src.preprocessing import make_splits


def test_three_distinct_algorithms(experiment):
    classes = {type(m.estimator).__name__ for m in experiment.models.values()}
    assert len(classes) == 3
    assert {"LogisticRegression", "RandomForestClassifier"} <= classes


def test_splits_are_disjoint_and_complete(experiment):
    s = experiment.split
    tr, va, te = set(s.train_idx), set(s.val_idx), set(s.test_idx)
    assert not (tr & va) and not (tr & te) and not (va & te)
    assert len(tr | va | te) == len(experiment.frame)


def test_preprocessing_fitted_on_training_rows_only(experiment):
    model = experiment.models["random_forest"]
    imputer = model.pipeline.named_steps["preprocess"].named_transformers_["num"].named_steps["impute"]
    cleaner = model.pipeline.named_steps["clean"]
    builder = model.pipeline.named_steps["features"]
    numeric = cleaner.numeric_columns_ + builder.features_
    train = builder.transform(cleaner.transform(experiment.X_split("train")))
    col = numeric.index("EXT_SOURCE_1")
    assert imputer.statistics_[col] == pytest.approx(train["EXT_SOURCE_1"].median())


def test_probability_outputs(experiment):
    X = experiment.X_split("test")
    for model in experiment.models.values():
        p = model.predict_pd(X)
        s = model.predict_score(X)
        assert p.shape == (len(X),) and np.all((p >= 0) & (p <= 1))
        assert np.all((s >= 0) & (s <= 1))


def test_calibrator_uses_validation_not_test(experiment):
    from src.calibration import ProbabilityCalibrator
    key = experiment.selected
    refit = ProbabilityCalibrator(experiment.config.calibration).fit(
        experiment.score_split("validation", key), experiment.y_split("validation"))
    np.testing.assert_allclose(refit.transform(experiment.score_split("test", key)), experiment.pd_split("test", key), rtol=1e-6)


def test_missing_values_and_unknown_categories_at_inference(experiment):
    model = experiment.selected_model
    row = pd.DataFrame({c: [np.nan] for c in model.input_columns})
    assert 0 <= model.predict_pd(row)[0] <= 1  # every value missing → imputed from training statistics
    row = experiment.X_split("test").iloc[[0]].copy()
    row["ORGANIZATION_TYPE"] = "A category never seen in training"
    row["NAME_CONTRACT_TYPE"] = "Crypto loans"
    row["AMT_INCOME_TOTAL"] = np.inf
    assert 0 <= model.predict_pd(row)[0] <= 1


def test_incompatible_inputs(experiment):
    model = experiment.selected_model
    partial = experiment.X_split("test").iloc[:3].drop(columns=["EXT_SOURCE_2", "AMT_CREDIT"])
    assert any("absent" in m for m in model.check_input(partial))
    assert len(model.predict_pd(partial)) == 3
    with pytest.raises(ValueError, match="incompatible"):
        model.predict_pd(pd.DataFrame({"unrelated": [1, 2]}))
    with pytest.raises(ValueError):
        model.predict_pd(pd.DataFrame(columns=model.input_columns))


def test_save_and_load_bundle(experiment, tmp_path):
    bundle = experiment.bundle()
    path = ps.save_bundle(bundle, "unit_test_model", tmp_path)
    loaded = ps.load_bundle(path, tmp_path)
    X = experiment.X_split("test").iloc[:20]
    np.testing.assert_allclose(loaded.model.predict_pd(X), bundle.model.predict_pd(X))
    assert loaded.threshold == bundle.threshold
    with pytest.raises(ValueError):
        ps.load_bundle(path, tmp_path / "elsewhere")  # only the models folder is trusted


def test_chronological_split_respects_time(synthetic_df):
    y = dv.apply_target_mapping(synthetic_df["TARGET"], [1], [0]).astype(int)
    s = make_splits(y, "chronological", 0.2, 0.2, dates=synthetic_df["APPLICATION_DATE"])
    d = pd.to_datetime(synthetic_df["APPLICATION_DATE"])
    assert d.iloc[s.train_idx].max() <= d.iloc[s.val_idx].min()
    assert d.iloc[s.val_idx].max() <= d.iloc[s.test_idx].min()
    with pytest.raises(ValueError):
        make_splits(y, "chronological", 0.2, 0.2, dates=None)


def test_small_or_single_class_data_errors():
    with pytest.raises(ValueError):
        make_splits(np.array([0, 1] * 10))  # fewer than 30 rows
    with pytest.raises(ValueError):
        make_splits(np.array([0] * 97 + [1] * 3), test_size=0.2, val_size=0.2)


def test_run_experiment_validation_errors(synthetic_df, mapping):
    with pytest.raises(ValueError, match="No predictor"):
        mdl.run_experiment(synthetic_df, {}, mapping, [], mdl.TrainingConfig())
    with pytest.raises(ValueError, match="target column"):
        mdl.run_experiment(synthetic_df, {}, mapping, ["TARGET"], mdl.TrainingConfig())
    with pytest.raises(ValueError, match="date column"):
        mdl.run_experiment(synthetic_df, {}, mapping, ["AMT_CREDIT"], mdl.TrainingConfig(split_strategy="chronological"))


def test_boosting_fallback_when_optional_libraries_absent(monkeypatch, synthetic_df, mapping, predictors):
    real = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec", lambda name, *a: None if name in ("lightgbm", "xgboost") else real(name, *a))
    assert mdl.boosting_backend() == "sklearn_hgb"
    assert "HistGradientBoosting" in mdl.model_label("gbm")
    cfg = mdl.TrainingConfig(model_keys=["gbm"], sample_size=2_000, params={"gbm": {"max_iter": 50}})
    res = mdl.run_experiment(synthetic_df, {}, mapping, predictors, cfg)
    assert type(res.models["gbm"].estimator).__name__ == "HistGradientBoostingClassifier"
    monkeypatch.setattr(xai, "shap_available", lambda: False)
    expl, meta = xai.local_explanation(res.models["gbm"], res.X_split("test").iloc[[0]], res.schema)
    assert not meta["exact"] and "Occlusion" in meta["method"] and len(expl)


def test_explanations_for_each_model(experiment):
    x = experiment.X_split("test").iloc[[0]]
    for key, model in experiment.models.items():
        expl, meta = xai.local_explanation(model, x, experiment.schema, experiment.bundle(key).transformed_mean, top=5)
        assert len(expl) == 5 and meta["method"]
        assert set(expl["feature"]) <= set(model.input_columns) | set(model.derived_features())
    lr = experiment.models["logreg"]
    contrib, meta = xai.contributions(lr, x, experiment.schema, experiment.bundle("logreg").transformed_mean)
    logit = np.log(lr.predict_score(x)[0] / (1 - lr.predict_score(x)[0]))
    assert contrib.iloc[0].sum() + meta["base_value"] == pytest.approx(logit, abs=1e-6)  # exact for linear models


def test_permutation_importance_and_pdp(experiment):
    model = experiment.selected_model
    pi = xai.permutation_importance_table(model, experiment.X_split("validation"), experiment.y_split("validation"),
                                          n_repeats=2, max_rows=800)
    assert set(pi["feature"]) == set(model.input_columns)
    curve = xai.partial_dependence_curve(model, experiment.X_split("validation"), "EXT_SOURCE_2", experiment.schema, 8, 100)
    assert len(curve) > 3 and curve["avg_pd"].iloc[0] > curve["avg_pd"].iloc[-1]  # higher external score → lower PD
