"""Scoring and decisions: model output → policy → explanation → audit record."""
from __future__ import annotations

import hashlib
import hmac
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from fastapi.exceptions import RequestValidationError
from pydantic import ValidationError

from creditiq_api.model_store import LoadedModel
from creditiq_core import explainability as xai
from creditiq_core._frames import iter_rows
from creditiq_core.finance.economics import expected_loss
from creditiq_core.formatting import val
from creditiq_core.jsonsafe import to_json
from creditiq_core.policy.engine import FLAGGED, Decision, apply_policy, decide
from creditiq_core.policy.schema import Policy
from creditiq_core.provenance import Provenance as P

TOP_REASONS = 8


def input_hash(inputs: dict[str, Any], key: bytes) -> str:
    """HMAC-SHA256 of the canonical JSON inputs. Lets an auditor match a known application without the log ever
    holding applicant data; without the key, the hash cannot be brute-forced from guessed inputs."""
    return hmac.new(key, to_json(inputs, sort_keys=True).encode(), hashlib.sha256).hexdigest()


def validate_applicant(model: LoadedModel, applicant: dict[str, Any], loc: tuple[str | int, ...]) -> dict[str, Any]:
    """Validate against the model's generated schema; errors use FastAPI's 422 format with the field location."""
    try:
        return model.request_model.model_validate(applicant).model_dump()
    except ValidationError as exc:
        errors = []
        for detail in exc.errors(include_url=False):
            e: dict[str, Any] = dict(detail)
            e["loc"] = (*loc, *e.get("loc", ()))
            if e.get("type") == "missing":
                e["msg"] = "Field required (send null if the value is unknown)"
            elif e.get("type") == "extra_forbidden":
                e["msg"] = "Not an input of the served model"
            e.pop("ctx", None)
            errors.append(e)
        raise RequestValidationError(errors) from exc


def input_warnings(model: LoadedModel, applicant: dict[str, Any]) -> list[str]:
    out = []
    for col, info in model.artifact.input_schema.items():
        v = applicant.get(col)
        if v is None:
            continue
        if info["kind"] == "numeric":
            if isinstance(v, float) and not math.isfinite(v):
                out.append(f"{col} is not finite and is treated as missing.")
            elif "p01" in info and (v < info["min"] or v > info["max"]):
                out.append(f"{col} = {val(v)} is outside the training range [{val(info['min'])}, {val(info['max'])}]; "
                           "the estimate extrapolates.")
        else:
            cats = info.get("categories", [])
            if info.get("n_categories", 0) <= len(cats) and str(v) not in cats:
                out.append(f"{col} = '{v}' was not seen in training; it is treated as an infrequent category.")
    return out


def caveats(model: LoadedModel) -> list[str]:
    out = []
    if model.is_synthetic:
        out.append("The model was trained on SYNTHETIC data; this estimate says nothing about real borrowers.")
    if not model.artifact.model.is_calibrated:
        out.append("Calibration is off: the PD is a model score, not a calibrated probability.")
    definition = model.metadata.get("target", {}).get("definition") or ""
    if "payment difficult" in definition.lower():
        out.append("The PD is the probability of the dataset's own event (an early payment difficulty); treating it "
                   "as a 12-month PD is an assumption.")
    out.append("Explanations show what the model associates with risk, not causes.")
    out.append("Educational decision-support prototype: outcomes are 'within threshold' or 'flagged for review', "
               "never an approval or a decline.")
    return out


def exposure_of(model: LoadedModel, applicant: dict[str, Any], override: float | None, policy: Policy) -> float:
    if override is not None:
        e = float(override)
    else:
        raw = applicant.get(model.exposure_col) if model.exposure_col else None
        e = float(raw) if raw is not None and math.isfinite(float(raw)) else policy.economics.default_exposure
    return max(e, 0.0) * policy.economics.ccf


def reasons(model: LoadedModel, applicant: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    expl, meta = xai.local_explanation(model.artifact.model, pd.DataFrame([applicant]), model.artifact.input_schema,
                                       model.artifact.transformed_mean, top=TOP_REASONS)
    rows = [{"feature": str(r.feature), "value": val(r.value), "contribution": float(r.contribution),
             "direction": "increases_risk" if r.contribution >= 0 else "decreases_risk", "derived": bool(r.derived)}
            for r in iter_rows(expl)]
    return rows, meta


def data_provenance(model: LoadedModel) -> str:
    return "synthetic" if model.is_synthetic else "observed"


def labels(model: LoadedModel, *kinds: P) -> list[P]:
    """Provenance labels for a number derived from the model's data (adds 'synthetic' for synthetic data)."""
    out = list(kinds)
    if model.is_synthetic and P.SYNTHETIC not in out:
        out.append(P.SYNTHETIC)
    return out


def score_provenance(model: LoadedModel, decide_mode: bool) -> dict[str, list[P]]:
    m = P.MODEL
    prov = {
        "pd": [m], "raw_score": [m], "risk_band": [m, P.ASSUMPTION], "threshold": [P.MEASURED, P.ASSUMPTION],
        "flagged_for_review": [m, P.ASSUMPTION], "ead": [P.OBSERVED, P.ASSUMPTION], "lgd": [P.ASSUMPTION],
        "expected_loss": [m, P.ASSUMPTION], "contribution": [m], "latency_ms": [P.MEASURED],
        "test_roc_auc": labels(model, P.MEASURED), "test_calibration_error": labels(model, P.MEASURED),
        "band_observed_default_rate": labels(model, P.OBSERVED), "band_mean_pd": labels(model, m),
        "band_n": labels(model, P.OBSERVED),
    }
    if decide_mode:
        prov.update({"value": [P.OBSERVED], "limit": [P.ASSUMPTION]})
    return prov


@dataclass
class Scored:
    applicant: dict[str, Any]
    pd: float
    raw_score: float
    decision: Decision
    ead: float
    expected_loss: float
    reasons: list[dict[str, Any]]
    explanation: dict[str, Any]
    warnings: list[str]


def score_one(model: LoadedModel, policy: Policy, applicant: dict[str, Any], exposure: float | None,
              with_rules: bool) -> Scored:
    raw = float(model.artifact.predict_score(applicant)[0])
    pd_value = float(model.artifact.model.calibrator.transform(np.array([raw]))[0])
    if with_rules:
        decision = decide(policy, applicant, pd_value, model.artifact.threshold)
    else:  # /score: thresholds and bands only; hard rules belong to /decide
        no_rules = policy.model_copy(update={"hard_rules": []})
        decision = decide(no_rules, applicant, pd_value, model.artifact.threshold)
    ead = exposure_of(model, applicant, exposure, policy)
    el = float(expected_loss(pd_value, policy.economics.lgd, ead))
    rs, meta = reasons(model, applicant)
    return Scored(applicant, pd_value, raw, decision, ead, el, rs, meta, input_warnings(model, applicant))


def uncertainty(model: LoadedModel, policy: Policy, band: str) -> dict[str, Any]:
    rel = model.band_reliability(tuple(policy.band_cutoffs), tuple(policy.band_labels)).get(band, {})
    return {"test_roc_auc": model.test_metrics.get("roc_auc"), "test_calibration_error": model.test_ece,
            "band_observed_default_rate": rel.get("observed_default_rate"), "band_mean_pd": rel.get("mean_pd"),
            "band_n": int(rel.get("n", 0))}


def score_frame(model: LoadedModel, policy: Policy, frame: pd.DataFrame) -> pd.DataFrame:
    """Vectorised batch scoring (thresholds and hard rules; no per-row explanations)."""
    X = frame.reindex(columns=model.input_columns)
    raw = model.artifact.predict_score(X)
    pd_values = model.artifact.model.calibrator.transform(raw)
    res = apply_policy(policy, X, pd_values, model.artifact.threshold)
    if model.exposure_col and model.exposure_col in X.columns:
        e = pd.to_numeric(X[model.exposure_col], errors="coerce").fillna(policy.economics.default_exposure)
        ead = e.clip(lower=0).to_numpy(dtype=float) * policy.economics.ccf
    else:
        ead = np.full(len(X), policy.economics.default_exposure * policy.economics.ccf)
    res["raw_score"] = raw
    res["ead"] = ead
    res["expected_loss"] = np.asarray(pd_values) * policy.economics.lgd * ead
    res["outcome"] = np.where(res["flagged"], FLAGGED, "within_threshold")
    fired_cols = [c for c in res.columns if c.startswith("rule:")]
    names = [c[5:] for c in fired_cols]
    hits = res[fired_cols].to_numpy(dtype=bool) if fired_cols else np.zeros((len(res), 0), dtype=bool)
    res["rules_fired"] = [[names[j] for j in np.flatnonzero(row)] for row in hits]
    return res
