"""Audit-log writes. Synchronous and in the request: if the write fails, the request fails (503), so no
decision is ever returned without an audit record. Inputs are stored only as an HMAC."""
from __future__ import annotations

import logging
import time
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException, status
from sqlalchemy import insert
from sqlalchemy.exc import SQLAlchemyError

from creditiq_api.db import Database, Decision, utcnow
from creditiq_api.model_store import LoadedModel
from creditiq_api.scoring import input_hash
from creditiq_api.security import Principal
from creditiq_core._frames import iter_rows
from creditiq_core.policy.schema import Policy

log = logging.getLogger(__name__)


def _fail(exc: Exception) -> HTTPException:
    log.error("audit write failed", extra={"error": f"{type(exc).__name__}: {exc}"})
    return HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE,
                         "The audit store is unavailable, so no decision was made. Try again later.")


def record_decision(app: FastAPI, db: Database, *, request_id: str, endpoint: str, principal: Principal,
                    model: LoadedModel, policy: Policy, pd: float, raw_score: float, band: str, outcome: str,
                    threshold: float, threshold_source: str, rules_fired: list[dict[str, Any]],
                    top_reasons: list[dict[str, Any]], ead: float, expected_loss: float, applicant: dict[str, Any],
                    latency_ms: float) -> None:
    settings = app.state.settings
    t0 = time.perf_counter()
    try:
        with db.session() as s:
            s.add(Decision(request_id=request_id, endpoint=endpoint, api_key_id=principal.key_id,
                           role=principal.role, model_name=model.name, model_version=model.version,
                           policy_version=policy.version, pd=pd, raw_score=raw_score, band=band, outcome=outcome,
                           threshold=threshold, threshold_source=threshold_source, rules_fired=rules_fired,
                           top_reasons=top_reasons, ead=ead, expected_loss=expected_loss,
                           input_hash=input_hash(applicant, settings.hash_key()),
                           hash_key_id=settings.audit_hash_key_id, latency_ms=latency_ms))
    except SQLAlchemyError as exc:
        raise _fail(exc) from exc
    app.state.metrics.audit_seconds.observe(time.perf_counter() - t0)


def record_decisions_bulk(app: FastAPI, db: Database, *, batch_id: str, principal: Principal, model: LoadedModel,
                          policy: Policy, request_ids: list[str], frame: pd.DataFrame,
                          applicants: list[dict[str, Any]]) -> None:
    settings = app.state.settings
    key = settings.hash_key()
    now = utcnow()
    rows = [{"request_id": rid, "created_at": now, "endpoint": "batch", "batch_id": batch_id,
             "api_key_id": principal.key_id, "role": principal.role, "model_name": model.name,
             "model_version": model.version, "policy_version": policy.version, "pd": float(r.pd),
             "raw_score": float(r.raw_score), "band": str(r.band), "outcome": str(r.outcome),
             "threshold": float(r.threshold), "threshold_source": str(r.threshold_source),
             "rules_fired": [{"id": rid_} for rid_ in r.rules_fired], "top_reasons": [], "ead": float(r.ead),
             "expected_loss": float(r.expected_loss), "input_hash": input_hash(a, key),
             "hash_key_id": settings.audit_hash_key_id, "latency_ms": None}
            for rid, r, a in zip(request_ids, iter_rows(frame, index=False), applicants, strict=True)]
    t0 = time.perf_counter()
    try:
        with db.session() as s:
            s.execute(insert(Decision), rows)
    except SQLAlchemyError as exc:
        raise _fail(exc) from exc
    app.state.metrics.audit_seconds.observe(time.perf_counter() - t0)
