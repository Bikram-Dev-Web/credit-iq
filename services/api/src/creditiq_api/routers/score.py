"""POST /v1/score, /v1/score/batch and /v1/decide."""
from __future__ import annotations

import csv
import io
import json
import time
from collections.abc import Iterator
from typing import Any, Literal

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import StreamingResponse

from creditiq_api import deps, scoring
from creditiq_api.audit import record_decision, record_decisions_bulk
from creditiq_api.db import Database
from creditiq_api.model_store import LoadedModel
from creditiq_api.ratelimit import score_rate_limit
from creditiq_api.schemas import COMMON_ERRORS, DecideResponse, ScoreRequest, ScoreResponse
from creditiq_api.security import Principal
from creditiq_core.data.csv import read_csv_file
from creditiq_core.jsonsafe import to_json
from creditiq_core.policy.schema import Policy

router = APIRouter(prefix="/v1", tags=["scoring"], responses=COMMON_ERRORS)


def _build(request: Request, model: LoadedModel, policy: Policy, body: ScoreRequest, principal: Principal,
           db: Database, decide_mode: bool) -> dict[str, Any]:
    t0 = time.perf_counter()
    applicant = scoring.validate_applicant(model, body.applicant, ("body", "applicant"))
    s = scoring.score_one(model, policy, applicant, body.exposure, with_rules=decide_mode)
    d = s.decision
    rid: str = request.state.request_id
    endpoint = "decide" if decide_mode else "score"
    latency_ms = (time.perf_counter() - t0) * 1000
    record_decision(request.app, db, request_id=rid, endpoint=endpoint, principal=principal, model=model,
                    policy=policy, pd=s.pd, raw_score=s.raw_score, band=d.band, outcome=d.outcome,
                    threshold=d.threshold, threshold_source=d.threshold_source,
                    rules_fired=[r.to_dict() for r in d.rules_fired],
                    top_reasons=[{"feature": r["feature"], "contribution": r["contribution"]} for r in s.reasons],
                    ead=s.ead, expected_loss=s.expected_loss, applicant=applicant, latency_ms=latency_ms)
    metrics = request.app.state.metrics
    metrics.pd.labels(endpoint).observe(s.pd)
    metrics.decisions.labels(endpoint, d.outcome).inc()
    for r in d.rules_fired:
        metrics.rules_fired.labels(r.id).inc()
    out: dict[str, Any] = {
        "request_id": rid, "model_name": model.name, "model_version": model.version,
        "policy_version": policy.version, "pd": s.pd, "raw_score": s.raw_score,
        "is_calibrated": model.artifact.model.is_calibrated, "risk_band": d.band, "threshold": d.threshold,
        "threshold_source": d.threshold_source, "flagged_for_review": d.flagged, "outcome": d.outcome,
        "ead": s.ead, "lgd": policy.economics.lgd, "expected_loss": s.expected_loss, "top_reasons": s.reasons,
        "explanation_method": s.explanation.get("method", ""), "explanation_units": s.explanation.get("units", ""),
        "uncertainty": scoring.uncertainty(model, policy, d.band), "warnings": s.warnings,
        "caveats": scoring.caveats(model), "latency_ms": round(latency_ms, 3),
        "data_provenance": scoring.data_provenance(model),
        "provenance": scoring.score_provenance(model, decide_mode),
    }
    if decide_mode:
        out.update({"policy_name": policy.name, "rules": [r.to_dict() for r in d.rules],
                    "rules_fired": [r.id for r in d.rules_fired], "reasons": d.reasons()})
    return out


@router.post("/score", response_model=ScoreResponse, summary="Score one applicant")
def score(request: Request, body: ScoreRequest, principal: Principal = Depends(score_rate_limit),
          model: LoadedModel = Depends(deps.model), policy: Policy = Depends(deps.policy),
          db: Database = Depends(deps.db)) -> dict[str, Any]:
    """Calibrated PD, risk band, threshold, the flag for review, expected loss and the top reasons (signed
    contributions, one-hot columns summed back to their source input). Uses the active policy's thresholds and
    bands; hard rules are applied by `/v1/decide`."""
    return _build(request, model, policy, body, principal, db, decide_mode=False)


@router.post("/decide", response_model=DecideResponse, summary="Score one applicant and apply the lending policy")
def decide(request: Request, body: ScoreRequest, principal: Principal = Depends(score_rate_limit),
           model: LoadedModel = Depends(deps.model), policy: Policy = Depends(deps.policy),
           db: Database = Depends(deps.db)) -> dict[str, Any]:
    """Everything `/v1/score` returns, plus every hard rule's result, which rules fired, and why."""
    return _build(request, model, policy, body, principal, db, decide_mode=True)


# ---------------------------------------------------------------------------- batch

async def _read_batch(request: Request, model: LoadedModel, max_rows: int, max_bytes: int) -> pd.DataFrame:
    declared = request.headers.get("content-length")
    if declared and declared.isdigit() and int(declared) > max_bytes:
        raise HTTPException(status.HTTP_413_CONTENT_TOO_LARGE, f"Batch body exceeds {max_bytes:,} bytes.")
    body = bytearray()
    async for chunk in request.stream():
        body.extend(chunk)
        if len(body) > max_bytes:
            raise HTTPException(status.HTTP_413_CONTENT_TOO_LARGE, f"Batch body exceeds {max_bytes:,} bytes.")
    ctype = request.headers.get("content-type", "").split(";")[0].strip().lower()
    if ctype in ("text/csv", "application/csv"):
        try:
            frame, _ = read_csv_file(bytes(body), name="batch.csv", optimize_memory=False)
        except ValueError as exc:
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, f"Unreadable CSV: {exc}") from exc
        frame = frame.astype(object).where(frame.notna(), None)
    elif ctype == "application/json":
        try:
            payload = json.loads(body)
        except json.JSONDecodeError as exc:
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, f"Invalid JSON: {exc}") from exc
        rows = payload.get("applicants") if isinstance(payload, dict) else None
        if not isinstance(rows, list) or not rows or not all(isinstance(r, dict) for r in rows):
            raise RequestValidationError([{"type": "value_error", "loc": ("body", "applicants"),
                                           "msg": "Expected {\"applicants\": [{...}, ...]} with at least one row",
                                           "input": None}])
        if len(rows) > max_rows:
            raise HTTPException(status.HTTP_413_CONTENT_TOO_LARGE, f"Batch has {len(rows):,} rows; the limit is "
                                                                   f"{max_rows:,}.")
        rows = [scoring.validate_applicant(model, row, ("body", "applicants", i)) for i, row in enumerate(rows)]
        frame = pd.DataFrame(rows, columns=model.input_columns)
    else:
        raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, "Send application/json or text/csv.")
    if len(frame) > max_rows:
        raise HTTPException(status.HTTP_413_CONTENT_TOO_LARGE, f"Batch has {len(frame):,} rows; the limit is "
                                                               f"{max_rows:,}.")
    cols = set(model.input_columns)
    errors = [{"type": "missing", "loc": ("body", col), "input": None,
               "msg": "Column required (leave cells empty for unknown values)"}
              for col in sorted(cols - set(frame.columns))]
    errors += [{"type": "extra_forbidden", "loc": ("body", col), "input": None, "msg": "Not an input of the served "
                                                                                       "model"}
               for col in sorted(set(frame.columns) - cols)]
    if errors:
        raise RequestValidationError(errors)
    return frame


BATCH_COLS = ["pd", "raw_score", "band", "threshold", "threshold_source", "outcome", "ead", "expected_loss"]


@router.post("/score/batch", summary="Score a batch (JSON or CSV); streams NDJSON or CSV",
             responses={200: {"content": {"application/x-ndjson": {}, "text/csv": {}},
                              "description": "A metadata line, then one result per input row in input order"},
                        413: {"description": "Too many rows or bytes"}},
             openapi_extra={"requestBody": {"required": True, "content": {
                 "application/json": {"schema": {"$ref": "#/components/schemas/BatchScoreRequest"}},
                 "text/csv": {"schema": {"type": "string"}}}}})
async def score_batch(request: Request, output: Literal["ndjson", "csv"] = Query("ndjson", alias="format"),
                      principal: Principal = Depends(score_rate_limit), model: LoadedModel = Depends(deps.model),
                      policy: Policy = Depends(deps.policy), db: Database = Depends(deps.db)) -> StreamingResponse:
    """Applies the active policy (thresholds and hard rules) to every row. Limits: `BATCH_MAX_ROWS` rows and
    `BATCH_MAX_BYTES` bytes. Every row is written to the audit log with a shared `batch_id`. Per-row explanations
    are not computed in batch mode; use `/v1/decide` for one applicant."""
    settings = request.app.state.settings
    frame = await _read_batch(request, model, settings.batch_max_rows, settings.batch_max_bytes)
    batch_id: str = request.state.request_id
    res = scoring.score_frame(model, policy, frame)
    applicants = [{str(k): v for k, v in row.items()}
                  for row in frame.reindex(columns=model.input_columns).to_dict(orient="records")]
    request_ids = [f"{batch_id}-{i}" for i in range(len(res))]
    record_decisions_bulk(request.app, db, batch_id=batch_id, principal=principal, model=model, policy=policy,
                          request_ids=request_ids, frame=res, applicants=applicants)
    metrics = request.app.state.metrics
    metrics.batch_rows.inc(len(res))
    for outcome, n in res["outcome"].value_counts().items():
        metrics.decisions.labels("batch", str(outcome)).inc(int(n))
    metrics.pd.labels("batch").observe(float(np.mean(res["pd"])))
    provenance = scoring.data_provenance(model)
    table = res[[*BATCH_COLS, "rules_fired"]]

    def records() -> Iterator[dict[str, Any]]:
        for i, (rid, row) in enumerate(zip(request_ids, table.itertuples(index=False), strict=True)):
            rec = {"row": i, "request_id": rid, **dict(zip([*BATCH_COLS, "rules_fired"], row, strict=True))}
            rec["flagged_for_review"] = rec["outcome"] == "flagged_for_review"
            yield rec

    headers = {"X-Batch-Id": batch_id, "X-Model-Version": model.version, "X-Policy-Version": policy.version,
               "X-Data-Provenance": provenance}
    if output == "csv":
        def csv_stream() -> Iterator[str]:
            buf = io.StringIO()
            w = csv.DictWriter(buf, fieldnames=["row", "request_id", *BATCH_COLS, "flagged_for_review", "rules_fired"])
            w.writeheader()
            for i, rec in enumerate(records()):
                rec["rules_fired"] = ";".join(rec["rules_fired"])
                w.writerow(rec)
                if i % 1000 == 999:
                    yield buf.getvalue()
                    buf.seek(0)
                    buf.truncate()
            yield buf.getvalue()

        return StreamingResponse(csv_stream(), media_type="text/csv", headers=headers)

    prov = scoring.score_provenance(model, True)

    def ndjson() -> Iterator[str]:
        yield to_json({"meta": {"batch_id": batch_id, "model_version": model.version,
                                "policy_version": policy.version, "rows": len(res), "data_provenance": provenance,
                                "provenance": {k: [str(p) for p in v] for k, v in prov.items()}}}) + "\n"
        chunk: list[str] = []
        for rec in records():
            chunk.append(to_json(rec))
            if len(chunk) == 1000:
                yield "\n".join(chunk) + "\n"
                chunk = []
        if chunk:
            yield "\n".join(chunk) + "\n"

    return StreamingResponse(ndjson(), media_type="application/x-ndjson", headers=headers)
