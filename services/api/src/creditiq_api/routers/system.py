"""Health, readiness, metrics, meta, overview and the decisions (audit) log."""
from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response, status
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from sqlalchemy import Integer, func, or_, select

from creditiq_api import __version__, deps, drift
from creditiq_api.db import Database, Decision
from creditiq_api.model_store import ModelHolder
from creditiq_api.schemas import COMMON_ERRORS, MetaOut, ResultOut
from creditiq_api.scoring import data_provenance, labels
from creditiq_api.security import Principal, analyst
from creditiq_core.finance import portfolio as pa
from creditiq_core.provenance import PROVENANCE_LABELS
from creditiq_core.provenance import Provenance as P

router = APIRouter(tags=["system"], responses=COMMON_ERRORS)


@router.get("/healthz", summary="Liveness")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/readyz", summary="Readiness: a model is loaded, a policy is active and the database answers")
def readyz(request: Request, response: Response) -> dict[str, Any]:
    holder: ModelHolder = request.app.state.holder
    checks = {"model": holder.current is not None, "database": request.app.state.db.ping()}
    try:
        request.app.state.policies.active()
        checks["policy"] = True
    except LookupError:
        checks["policy"] = False
    ok = all(checks.values())
    if not ok:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return {"status": "ready" if ok else "not ready", "checks": checks,
            "model_version": holder.current.version if holder.current else None, "model_error": holder.last_error}


@router.get("/metrics", include_in_schema=False)
def metrics(request: Request) -> Response:
    return Response(generate_latest(request.app.state.metrics.registry), media_type=CONTENT_TYPE_LATEST)


@router.get("/v1/meta", response_model=MetaOut, tags=["system"], summary="Versions, environment and caller role")
def meta(request: Request, principal: Principal = Depends(analyst)) -> dict[str, Any]:
    s = request.app.state.settings
    m = request.app.state.holder.current
    try:
        policy = request.app.state.policies.active()
    except LookupError:
        policy = None
    return {"api_version": __version__, "environment": s.environment, "role": principal.role,
            "admin_actions_enabled": s.allow_admin_actions, "model_name": s.model_name,
            "model_version": m.version if m else None, "model_label": m.label if m else None,
            "model_loaded_at": m.loaded_at if m else None, "policy_version": policy.version if policy else None,
            "policy_name": policy.name if policy else None, "grafana_url": s.grafana_url,
            "mlflow_ui_url": s.mlflow_ui_url,
            "provenance_labels": {p.value: list(v) for p, v in PROVENANCE_LABELS.items()},
            "data_provenance": data_provenance(m) if m else "synthetic", "provenance": {}}


@router.get("/v1/overview", response_model=ResultOut, tags=["system"], summary="Dashboard KPIs, trends and alerts")
def overview(request: Request, days: int = Query(14, ge=1, le=90), _: Principal = Depends(analyst),
             db: Database = Depends(deps.db)) -> dict[str, Any]:
    model = deps.model(request)
    policy = deps.policy(request)
    test = model.snapshot.split("test")
    port = pa.build_portfolio(test.drop(columns=["split", "y", "pd", "raw_score"]), test["y"], test["pd"],
                              model.exposure_col, policy.economics.lgd, policy.economics.ccf,
                              policy.economics.default_exposure, policy.band_cutoffs)
    kpis = pa.portfolio_kpis(port)
    from creditiq_core.policy.engine import apply_policy

    flag_rate_test = float(apply_policy(policy, test, test["pd"].to_numpy(), model.artifact.threshold)["flagged"].mean())
    since = datetime.now(UTC) - timedelta(days=days)
    with db.session() as s:
        day = func.date(Decision.created_at)
        rows = s.execute(select(day.label("day"), func.count(), func.avg(Decision.pd),
                                func.sum((Decision.outcome == "flagged_for_review").cast(Integer)))
                         .where(Decision.created_at >= since).group_by(day).order_by(day)).all()
        total = s.scalar(select(func.count()).select_from(Decision)) or 0
    trend = [{"day": str(d), "decisions": int(n), "avg_pd": float(a or 0), "flag_rate": float(f or 0) / n if n
              else None} for d, n, a, f in rows]
    latest = drift.run_to_dict(drift.fetch_run(db), with_detail=False)
    result = {
        "model_version": model.version, "model_label": model.label, "policy_version": policy.version,
        "portfolio_ead": kpis["total_ead"], "expected_loss_12m": kpis["total_el"], "el_rate": kpis["el_rate"],
        "observed_default_rate": kpis["observed_default_rate"], "avg_pd": kpis["avg_pd"],
        "flag_rate_test": flag_rate_test, "test_roc_auc": model.test_metrics.get("roc_auc"),
        "test_pr_auc": model.test_metrics.get("pr_auc"), "n_loans": kpis["n_loans"],
        "decisions_total": int(total), "trend": trend,
        "drift_status": latest.get("status", "no checks yet") if latest else "no checks yet",
        "drift_checked_at": latest.get("created_at") if latest else None,
        "recent_alerts": latest.get("alerts", [])[:5] if latest else [],
    }
    obs, mod, sim = labels(model, P.OBSERVED), labels(model, P.MODEL), labels(model, P.MODEL, P.ASSUMPTION)
    prov = {"portfolio_ead": labels(model, P.OBSERVED, P.ASSUMPTION), "expected_loss_12m": sim, "el_rate": sim,
            "observed_default_rate": obs, "avg_pd": mod, "flag_rate_test": labels(model, P.MEASURED, P.ASSUMPTION),
            "test_roc_auc": labels(model, P.MEASURED), "test_pr_auc": labels(model, P.MEASURED), "n_loans": obs,
            "decisions_total": [P.OBSERVED], "decisions": [P.OBSERVED], "flag_rate": [P.OBSERVED, P.MODEL],
            "value": [P.MEASURED], "threshold": [P.ASSUMPTION], "id": [P.OBSERVED]}
    return {"result": result, "data_provenance": data_provenance(model), "provenance": prov}


# ---------------------------------------------------------------------------- decisions log

def _decision(d: Decision) -> dict[str, Any]:
    return {c: getattr(d, c) for c in (
        "request_id", "created_at", "endpoint", "batch_id", "api_key_id", "role", "model_name", "model_version",
        "policy_version", "pd", "raw_score", "band", "outcome", "threshold", "threshold_source", "rules_fired",
        "top_reasons", "ead", "expected_loss", "input_hash", "hash_key_id", "latency_ms")}


DECISION_PROV = {"pd": [P.MODEL], "raw_score": [P.MODEL], "threshold": [P.MEASURED, P.ASSUMPTION],
                 "ead": [P.OBSERVED, P.ASSUMPTION], "expected_loss": [P.MODEL, P.ASSUMPTION],
                 "contribution": [P.MODEL], "latency_ms": [P.MEASURED], "total": [P.OBSERVED],
                 "value": [P.OBSERVED], "limit": [P.ASSUMPTION]}


@router.get("/v1/decisions", response_model=ResultOut, tags=["decisions"], summary="Search the audit log")
def list_decisions(q: str | None = Query(None, max_length=64, description="request id, batch id or input hash "
                                                                          "prefix"),
                   outcome: Literal["within_threshold", "flagged_for_review"] | None = None,
                   band: str | None = None, endpoint: Literal["score", "decide", "batch"] | None = None,
                   model_version: str | None = None, limit: int = Query(50, ge=1, le=500),
                   offset: int = Query(0, ge=0), _: Principal = Depends(analyst),
                   db: Database = Depends(deps.db)) -> dict[str, Any]:
    """Newest first. Holds scores, outcomes, rules fired and an HMAC of the inputs — never the inputs."""
    stmt = select(Decision)
    if q:
        stmt = stmt.where(or_(Decision.request_id.startswith(q), Decision.batch_id.startswith(q),
                              Decision.input_hash.startswith(q)))
    for col, v in ((Decision.outcome, outcome), (Decision.band, band), (Decision.endpoint, endpoint),
                   (Decision.model_version, model_version)):
        if v is not None:
            stmt = stmt.where(col == v)
    with db.session() as s:
        total = s.scalar(select(func.count()).select_from(stmt.subquery())) or 0
        rows = s.scalars(stmt.order_by(Decision.id.desc()).limit(limit).offset(offset)).all()
        items = [_decision(d) for d in rows]
    return {"result": {"total": int(total), "limit": limit, "offset": offset, "items": items},
            "data_provenance": "observed", "provenance": DECISION_PROV}


@router.get("/v1/decisions/{request_id}", response_model=ResultOut, tags=["decisions"], summary="One decision")
def get_decision(request_id: str, _: Principal = Depends(analyst), db: Database = Depends(deps.db)) -> dict[str, Any]:
    with db.session() as s:
        d = s.scalar(select(Decision).where(Decision.request_id == request_id))
        if d is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, f"No decision with request id {request_id}.")
        item = _decision(d)
    return {"result": item, "data_provenance": "observed", "provenance": DECISION_PROV}
