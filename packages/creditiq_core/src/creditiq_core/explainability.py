"""Global and local explanations of the model score.

Methods, chosen by what is genuinely available:

* Logistic Regression – exact linear contributions: coef × (x − training mean)
  in log-odds of the model score.
* LightGBM / XGBoost – exact TreeSHAP values from the library's own
  ``pred_contrib`` routine (no ``shap`` package needed), in log-odds.
* Random Forest / sklearn HistGradientBoosting – TreeSHAP via the optional
  ``shap`` package when installed.
* Fallback (any model) – feature occlusion: change in the model score when one
  input is reset to its training median/mode. Approximate; contributions do not
  sum to the prediction.

One-hot and missing-indicator columns are summed back to their source column.
Attributions describe *what the model uses*, i.e. predictive association in the
training data — not causal effects. With correlated inputs, credit can be split
between features somewhat arbitrarily, so individual attributions can be unstable.
"""
from __future__ import annotations

import importlib.util
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score

from creditiq_core.models.credit_model import CreditModel


def shap_available() -> bool:
    return importlib.util.find_spec("shap") is not None


def _aggregate(matrix: np.ndarray, names: list[str], origin: dict[str, str]) -> pd.DataFrame:
    """Sum transformed-column contributions into source columns. matrix: (n_rows, n_transformed)."""
    df = pd.DataFrame(matrix, columns=names)
    groups = [origin.get(n, n) for n in names]
    return df.T.groupby(groups, sort=False).sum().T


def _baseline_row(schema: dict[str, Any], columns: list[str]) -> dict[str, Any]:
    base = {}
    for c in columns:
        info = schema.get(c, {})
        base[c] = info.get("median", np.nan) if info.get("kind") == "numeric" else info.get("mode", np.nan)
    return base


def _occlusion(model: CreditModel, X: pd.DataFrame, schema: dict[str, Any]) -> pd.DataFrame:
    base_score = model.predict_score(X)
    baseline = _baseline_row(schema, model.input_columns)
    out = {}
    for col in model.input_columns:
        X_mod = X.copy()
        X_mod[col] = baseline.get(col, np.nan)
        out[col] = base_score - model.predict_score(X_mod)
    return pd.DataFrame(out, index=X.index)


def contributions(model: CreditModel, X: pd.DataFrame, schema: dict[str, Any],
                  transformed_mean: np.ndarray | None = None,
                  allow_shap: bool = True) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Per-row, per-source-feature contributions plus metadata (method, units, base value)."""
    est = model.estimator
    meta: dict[str, Any] = {"method": "", "units": "", "base_value": None, "exact": False}
    names = model.transformed_feature_names()
    origin = model.feature_origin_map()

    if model.key == "logreg":
        Xt = np.asarray(model.transform(X), dtype=float)
        coef = est.coef_.ravel()
        mean = transformed_mean if transformed_mean is not None and len(transformed_mean) == len(coef) else np.zeros_like(coef)
        contrib = (Xt - mean) * coef
        meta.update(method="Exact linear contribution: coefficient × (value − training mean)",
                    units="log-odds of the model score", base_value=float(est.intercept_[0] + coef @ mean), exact=True)
        return _aggregate(contrib, names, origin), meta

    if model.backend == "lightgbm" and model.key == "gbm":
        Xt = np.asarray(model.transform(X), dtype=float)
        raw = est.booster_.predict(Xt, pred_contrib=True)
        meta.update(method="TreeSHAP (LightGBM native pred_contrib)", units="log-odds of the model score",
                    base_value=float(raw[0, -1]), exact=True)
        return _aggregate(raw[:, :-1], names, origin), meta

    if model.backend == "xgboost" and model.key == "gbm":
        import xgboost as xgb
        Xt = np.asarray(model.transform(X), dtype=float)
        raw = est.get_booster().predict(xgb.DMatrix(Xt), pred_contribs=True)
        meta.update(method="TreeSHAP (XGBoost native pred_contribs)", units="log-odds of the model score",
                    base_value=float(raw[0, -1]), exact=True)
        return _aggregate(raw[:, :-1], names, origin), meta

    if allow_shap and shap_available():
        try:
            import shap
            Xt = np.asarray(model.transform(X), dtype=float)
            explainer = shap.TreeExplainer(est)
            sv = explainer.shap_values(Xt, check_additivity=False)
            ev = explainer.expected_value
            if isinstance(sv, list):
                sv, ev = sv[1], ev[1]
            elif np.ndim(sv) == 3:
                sv = sv[:, :, 1]
                ev = np.ravel(ev)[1] if np.size(ev) > 1 else ev
            units = "probability (model score)" if model.key == "random_forest" else "log-odds of the model score"
            meta.update(method="TreeSHAP (shap package)", units=units, base_value=float(np.ravel(ev)[0]), exact=True)
            return _aggregate(np.asarray(sv), names, origin), meta
        except Exception as exc:  # fall back to occlusion and say why
            meta["fallback_reason"] = f"SHAP failed ({type(exc).__name__}: {exc}); used occlusion instead."
    elif not shap_available():
        meta["fallback_reason"] = "Optional 'shap' package not installed; used occlusion instead."

    occ = _occlusion(model, X, schema)
    meta.update(method="Occlusion: score change when the feature is reset to its training median/mode",
                units="probability (model score)", base_value=None, exact=False)
    return occ, meta


def local_explanation(model: CreditModel, x_row: pd.DataFrame, schema: dict[str, Any],
                      transformed_mean: np.ndarray | None = None,
                      top: int = 12) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Top contributing features for one applicant, with the value the model saw."""
    contrib, meta = contributions(model, x_row, schema, transformed_mean)
    series = contrib.iloc[0]
    clean_feat = model.pipeline[:2].transform(x_row).iloc[0]
    rows = []
    for feat, val in series.items():
        shown = clean_feat.get(feat, np.nan) if feat in clean_feat.index else np.nan
        rows.append({"feature": feat, "value": shown, "contribution": float(val),
                     "derived": feat in model.derived_features()})
    out = pd.DataFrame(rows)
    out["abs"] = out["contribution"].abs()
    out = out.sort_values("abs", ascending=False).drop(columns="abs")
    meta["n_features"] = len(out)
    return out.head(top).reset_index(drop=True), meta


def global_contribution_importance(model: CreditModel, X: pd.DataFrame, schema: dict[str, Any],
                                   transformed_mean: np.ndarray | None = None, max_rows: int = 400,
                                   seed: int = 42) -> tuple[pd.DataFrame, dict[str, Any]]:
    sample = X.sample(min(max_rows, len(X)), random_state=seed) if len(X) > max_rows else X
    contrib, meta = contributions(model, sample, schema, transformed_mean)
    imp = contrib.abs().mean().sort_values(ascending=False)
    mean_signed = contrib.mean()
    out = pd.DataFrame({"feature": imp.index, "mean_abs_contribution": imp.values,
                        "mean_signed_contribution": mean_signed.reindex(imp.index).values})
    meta["rows_used"] = len(sample)
    return out, meta


def permutation_importance_table(model: CreditModel, X: pd.DataFrame, y: ArrayLike, n_repeats: int = 3,
                                 max_rows: int = 3000, seed: int = 42) -> pd.DataFrame:
    """Drop in validation ROC-AUC when one raw input is shuffled (model-agnostic)."""
    y = np.asarray(y).astype(int)
    if len(np.unique(y)) < 2:
        raise ValueError("Permutation importance needs both classes in the evaluation data.")
    if len(X) > max_rows:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(X), max_rows, replace=False)
        X, y = X.iloc[idx], y[idx]
    res = permutation_importance(model.pipeline, X, y, scoring="roc_auc", n_repeats=n_repeats,
                                 random_state=seed, n_jobs=1)
    out = pd.DataFrame({"feature": X.columns, "auc_drop_mean": res.importances_mean, "auc_drop_std": res.importances_std})
    out["baseline_auc"] = roc_auc_score(y, model.predict_score(X))
    return out.sort_values("auc_drop_mean", ascending=False).reset_index(drop=True)


def native_importance(model: CreditModel, transformed_mean: np.ndarray | None = None) -> tuple[pd.DataFrame, str]:
    """Model-internal importance aggregated to source features."""
    est = model.estimator
    names = model.transformed_feature_names()
    origin = model.feature_origin_map()
    if model.key == "logreg":
        coef = est.coef_.ravel()
        if transformed_mean is not None and len(transformed_mean) == len(coef):
            # Standard deviation of each source feature's log-odds contribution:
            # one-hot columns of one feature are mutually exclusive → Var = Σ c²p − (Σ cp)²;
            # standardised numeric columns (incl. missing indicators) have unit variance → Var ≈ Σ c².
            m = np.clip(np.asarray(transformed_mean, dtype=float), 0, 1)
            groups: dict[str, float] = {}
            frame = pd.DataFrame({"name": names, "coef": coef, "p": m, "src": [origin.get(n, n) for n in names]})
            for src, g in frame.groupby("src", sort=False):
                cat = g["name"].str.startswith("cat__")
                var = float((g.loc[~cat, "coef"] ** 2).sum())
                if cat.any():
                    c, p_ = g.loc[cat, "coef"], g.loc[cat, "p"]
                    var += float((c ** 2 * p_).sum() - (c * p_).sum() ** 2)
                groups[str(src)] = float(np.sqrt(max(var, 0.0)))
            agg = pd.Series(groups).sort_values(ascending=False)
            total = agg.sum()
            imp = agg.to_numpy(dtype=float)
            return (pd.DataFrame({"feature": agg.index, "importance": imp / total if total else imp}),
                    "Standard deviation of each input's log-odds contribution across training applicants")
        vals, method = np.abs(coef), "|coefficient| on standardised inputs (summed over encoded columns)"
    elif hasattr(est, "feature_importances_"):
        vals = np.asarray(est.feature_importances_, dtype=float)
        method = "Impurity-based importance (biased toward high-cardinality features)" if model.key == "random_forest" else "Split-based importance from the booster"
    else:
        return pd.DataFrame(columns=["feature", "importance"]), "Not available for this estimator"
    agg = _aggregate(vals.reshape(1, -1), names, origin).iloc[0].sort_values(ascending=False)
    total = agg.sum()
    imp = agg.to_numpy(dtype=float)
    return pd.DataFrame({"feature": agg.index, "importance": imp / total if total else imp}), method


def partial_dependence_curve(model: CreditModel, X: pd.DataFrame, feature: str, schema: dict[str, Any],
                             grid_points: int = 20, max_rows: int = 300, seed: int = 42) -> pd.DataFrame:
    """Average calibrated PD when ``feature`` is set to each grid value for a sample of applicants."""
    if feature not in X.columns:
        raise KeyError(feature)
    sample = X.sample(min(max_rows, len(X)), random_state=seed) if len(X) > max_rows else X.copy()
    info = schema.get(feature, {})
    if info.get("kind") == "numeric":
        vals = pd.to_numeric(X[feature], errors="coerce").dropna()
        if vals.empty:
            return pd.DataFrame()
        grid = np.unique(np.quantile(vals, np.linspace(0.05, 0.95, grid_points)))
    else:
        grid = info.get("categories", [])[:15]
    rows = []
    for v in grid:
        X_mod = sample.copy()
        X_mod[feature] = v
        p = model.predict_pd(X_mod)
        rows.append({"value": v, "avg_pd": float(np.mean(p)), "p25": float(np.quantile(p, 0.25)), "p75": float(np.quantile(p, 0.75))})
    return pd.DataFrame(rows)
