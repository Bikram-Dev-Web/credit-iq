"""Streamlit session-state helpers.

All cross-page state lives in ``st.session_state`` under the keys below. Setting
an upstream item (dataset → target → features → experiment) clears everything
downstream so results from one dataset/target/configuration are never shown
for another.
"""
from __future__ import annotations

import streamlit as st

from . import config

DEFAULT_FIN = {
    "lgd": config.DEFAULT_LGD,
    "ccf": config.DEFAULT_CCF,
    "annual_rate": config.DEFAULT_ANNUAL_RATE,
    "funding_rate": config.DEFAULT_FUNDING_RATE,
    "opex_per_loan": config.DEFAULT_OPEX_PER_LOAN,
    "horizon_years": 1.0,
    "default_exposure": 500_000.0,
    "currency": "",
}

_DEFAULTS = {
    "df": None,
    "ds_info": None,
    "target": None,  # TargetMapping
    "roles": None,  # DataFrame of suggested/edited roles
    "predictors": None,  # list[str] once confirmed
    "excluded": {},
    "date_col": None,
    "exposure_col": None,
    "sensitive_cols": [],
    "leakage_table": None,
    "experiment": None,  # ExperimentResult
    "active_model_key": None,
    "loaded_bundle": None,  # ModelBundle loaded from disk
    "use_loaded_bundle": False,
    "threshold_override": {},
    "band_cutoffs": list(config.DEFAULT_BAND_CUTOFFS),
    "fin": dict(DEFAULT_FIN),
    "policy_table": None,
    "scenario_table": None,
    "fairness_table": None,
    "fairness_attr": None,
    "perm_importance": None,
    "proxy_table": None,
    "last_assessment": None,
    "monitor_result": None,
    "contribution_result": None,
    "global_contrib": None,
    "_bundle_cache": {},
}

_DOWNSTREAM = {
    "dataset": ["target", "roles", "predictors", "excluded", "date_col", "exposure_col", "sensitive_cols",
                "leakage_table", "experiment", "active_model_key", "threshold_override", "policy_table",
                "scenario_table", "fairness_table", "fairness_attr", "perm_importance", "proxy_table", "last_assessment", "monitor_result", "contribution_result",
                "global_contrib", "_bundle_cache"],
    "target": ["roles", "predictors", "excluded", "leakage_table", "experiment", "active_model_key",
               "threshold_override", "policy_table", "scenario_table", "fairness_table", "perm_importance",
               "proxy_table", "last_assessment", "monitor_result", "contribution_result", "global_contrib", "_bundle_cache"],
    "features": ["experiment", "active_model_key", "threshold_override", "policy_table", "scenario_table",
                 "fairness_table", "perm_importance", "proxy_table", "last_assessment", "monitor_result",
                 "contribution_result", "global_contrib", "_bundle_cache"],
    "experiment": ["threshold_override", "policy_table", "scenario_table", "fairness_table", "fairness_attr",
                   "perm_importance", "proxy_table", "last_assessment", "monitor_result", "contribution_result",
                   "global_contrib", "_bundle_cache"],
    # switching the active model (or loading a bundle) invalidates results computed with the previous model
    "model": ["policy_table", "scenario_table", "fairness_table", "fairness_attr", "proxy_table", "last_assessment",
              "contribution_result", "global_contrib"],
}


def init_state() -> None:
    for key, value in _DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = value.copy() if isinstance(value, (dict, list)) else value


def _reset(level: str) -> None:
    for key in _DOWNSTREAM[level]:
        default = _DEFAULTS[key]
        st.session_state[key] = default.copy() if isinstance(default, (dict, list)) else default


def set_dataset(df, info) -> None:
    _reset("dataset")
    st.session_state.df = df
    st.session_state.ds_info = info


def set_target(mapping) -> None:
    _reset("target")
    st.session_state.target = mapping


def set_features(predictors, excluded, date_col, exposure_col, sensitive_cols, roles) -> None:
    _reset("features")
    st.session_state.predictors = list(predictors)
    st.session_state.excluded = dict(excluded)
    st.session_state.date_col = date_col
    st.session_state.exposure_col = exposure_col
    st.session_state.sensitive_cols = list(sensitive_cols)
    st.session_state.roles = roles


def set_experiment(result) -> None:
    _reset("experiment")
    st.session_state.experiment = result
    st.session_state.active_model_key = result.selected
    st.session_state.use_loaded_bundle = False


def set_active_model(key: str | None = None, loaded_bundle=None) -> None:
    """Make a trained model (by key) or a loaded bundle the active model."""
    _reset("model")
    if loaded_bundle is not None:
        st.session_state.loaded_bundle = loaded_bundle
        st.session_state.use_loaded_bundle = True
    else:
        st.session_state.active_model_key = key
        st.session_state.use_loaded_bundle = False


def has_data() -> bool:
    return st.session_state.get("df") is not None


def has_target() -> bool:
    return st.session_state.get("target") is not None


def has_features() -> bool:
    return bool(st.session_state.get("predictors"))


def has_experiment() -> bool:
    return st.session_state.get("experiment") is not None


def active_bundle():
    """Bundle for the active model: from the current experiment, or one loaded from disk."""
    if st.session_state.get("use_loaded_bundle") and st.session_state.get("loaded_bundle") is not None:
        return st.session_state.loaded_bundle
    exp = st.session_state.get("experiment")
    if exp is None:
        return st.session_state.get("loaded_bundle")
    key = st.session_state.get("active_model_key") or exp.selected
    cache = st.session_state.setdefault("_bundle_cache", {})
    cache_key = (exp.experiment_id, key)
    if cache_key not in cache:
        cache[cache_key] = exp.bundle(key)
    return cache[cache_key]


def experiment_bundle():
    """Bundle of the session experiment's active model — even while a bundle loaded from disk is the active model."""
    exp = st.session_state.get("experiment")
    key = experiment_model_key()
    if exp is None or key is None:
        return None
    cache = st.session_state.setdefault("_bundle_cache", {})
    cache_key = (exp.experiment_id, key)
    if cache_key not in cache:
        cache[cache_key] = exp.bundle(key)
    return cache[cache_key]


def has_model() -> bool:
    return active_bundle() is not None


def active_threshold() -> float | None:
    bundle = active_bundle()
    if bundle is None:
        return None
    key = (bundle.metadata.get("experiment_id"), bundle.model.key)
    return float(st.session_state.threshold_override.get(key, bundle.threshold))


def experiment_model_key():
    """Key of the session experiment's active model (pages that need labelled split data use this, not a loaded bundle)."""
    exp = st.session_state.get("experiment")
    if exp is None:
        return None
    return st.session_state.get("active_model_key") or exp.selected


def experiment_threshold() -> float | None:
    """Decision threshold for the session experiment's active model (validation-chosen, or the user's override).

    Pages that score the session experiment's splits must use this rather than :func:`active_threshold`, which follows a
    bundle loaded from disk when one is active — otherwise one model's threshold would be applied to another model's PDs.
    """
    exp = st.session_state.get("experiment")
    key = experiment_model_key()
    if exp is None or key is None:
        return None
    return float(st.session_state.threshold_override.get((exp.experiment_id, key), exp.thresholds[key]))


def set_threshold(value: float, experiment_id: str | None = None, model_key: str | None = None) -> None:
    """Override the decision threshold of a model (default: the active bundle) and drop results computed with the old one."""
    if experiment_id is None or model_key is None:
        bundle = active_bundle()
        if bundle is None:
            return
        experiment_id, model_key = bundle.metadata.get("experiment_id"), bundle.model.key
    st.session_state.threshold_override[(experiment_id, model_key)] = float(value)
    _reset("model")


def fin() -> dict:
    return st.session_state.fin


def currency() -> str:
    return st.session_state.fin.get("currency", "")
