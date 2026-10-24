"""FastAPI application factory and the ``creditiq-api`` entry point."""
from __future__ import annotations

import logging
import os
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from fastapi.responses import JSONResponse

from creditiq_api import __version__, drift, migrate
from creditiq_api.db import Database
from creditiq_api.model_store import LoadedModel, ModelHolder
from creditiq_api.observability import (
    DriftCollector,
    Metrics,
    RequestContextMiddleware,
    configure_logging,
)
from creditiq_api.policy_store import PolicyStore
from creditiq_api.ratelimit import RateLimiter
from creditiq_api.registry import FileRegistry, MlflowRegistry, Registry
from creditiq_api.routers import analytics, models, monitoring, policies, score, system
from creditiq_api.schemas import BatchScoreRequest
from creditiq_api.settings import Settings, get_settings
from creditiq_core.jsonsafe import to_json

log = logging.getLogger("creditiq.api")


class SafeJSONResponse(JSONResponse):
    """JSON where NaN and ±inf become null (undefined metrics are reported as null, never as invalid JSON)."""

    def render(self, content: Any) -> bytes:
        return to_json(content).encode("utf-8")

DESCRIPTION = """
CreditIQ scoring, decision and analytics API.

* **Auth:** send an API key in `X-API-Key`. `analyst` keys read and score; `admin` keys can also promote models
  and save or activate policies.
* **Outcomes** are `within_threshold` or `flagged_for_review` — never approve/deny.
* **Provenance:** every response has a `provenance` map (field → observed / measured / model / assumption /
  simulated / synthetic) and a `data_provenance` flag.
* **Privacy:** the audit log stores scores, outcomes, rules fired and an HMAC of the inputs, never the inputs.

*Educational decision-support prototype, not for real credit decisions.*
"""


def make_registry(settings: Settings) -> Registry:
    if settings.registry_backend == "mlflow":
        return MlflowRegistry(settings.mlflow_tracking_uri, settings.model_name, settings.model_cache_dir)
    return FileRegistry(settings.file_registry_path, settings.model_name)


def create_app(settings: Settings | None = None, registry: Registry | None = None) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings.log_level, settings.log_json)
    if settings.database_url.startswith("sqlite:///"):
        Path(settings.database_url.removeprefix("sqlite:///")).parent.mkdir(parents=True, exist_ok=True)
    if settings.auto_migrate:
        migrate.upgrade(settings.database_url)
    settings.hash_key()  # fail fast when the audit key is required but missing
    db = Database(settings.database_url)
    metrics = Metrics()
    holder = ModelHolder(registry or make_registry(settings), settings.model_alias, settings.registry_poll_seconds)
    holder.listeners.append(lambda m: metrics.set_model(m.name, m.version))
    store = PolicyStore(db, settings.policies_dir, settings.active_policy)

    _drift_cache: dict[str, Any] = {"at": 0.0, "run": None}

    def latest_run() -> dict[str, Any] | None:
        if time.monotonic() - _drift_cache["at"] > 15:
            _drift_cache["run"] = drift.run_to_dict(drift.fetch_run(db)) or None
            _drift_cache["at"] = time.monotonic()
        run: dict[str, Any] | None = _drift_cache["run"]
        return run

    metrics.registry.register(DriftCollector(latest_run))

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        store.seed()
        metrics.set_policy(store.active().version)
        holder.refresh("startup")
        if holder.current is None:
            log.error("no model loaded at start-up", extra={"error": holder.last_error})
        holder.start_polling()
        yield
        holder.stop()
        db.engine.dispose()

    app = FastAPI(title="CreditIQ API", version=__version__, description=DESCRIPTION, lifespan=lifespan,
                  default_response_class=SafeJSONResponse,
                  openapi_tags=[{"name": t} for t in ("scoring", "models", "policies", "portfolio", "scenarios",
                                                      "fairness", "monitoring", "decisions", "system")])
    app.state.settings = settings
    app.state.db = db
    app.state.metrics = metrics
    app.state.holder = holder
    app.state.policies = store
    app.state.version_cache = {}
    app.state.api_keys = settings.parsed_api_keys()
    app.state.score_limiter = RateLimiter(settings.rate_limit_score)
    if not app.state.api_keys:
        log.warning("API_KEYS is empty: every authenticated endpoint will return 401.")

    app.add_middleware(RequestContextMiddleware)
    app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origin_list, allow_credentials=False,
                       allow_methods=["GET", "POST"], allow_headers=["X-API-Key", "Content-Type", "X-Request-ID"],
                       expose_headers=["X-Request-ID", "X-Batch-Id", "X-Model-Version", "X-Policy-Version"])

    for r in (system.router, score.router, models.router, policies.router, analytics.router, monitoring.router):
        app.include_router(r)

    @app.exception_handler(HTTPException)
    async def http_error(request: Request, exc: HTTPException) -> JSONResponse:
        detail = exc.detail if isinstance(exc.detail, dict) else {"message": exc.detail}
        return SafeJSONResponse({"error": detail, "request_id": getattr(request.state, "request_id", None)},
                            status_code=exc.status_code, headers=exc.headers)

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        errors = [{"loc": list(e.get("loc", ())), "msg": e.get("msg"), "type": e.get("type")} for e in exc.errors()]
        return SafeJSONResponse({"error": {"message": "Request validation failed", "errors": errors},
                             "detail": errors, "request_id": getattr(request.state, "request_id", None)},
                            status_code=422)

    @app.exception_handler(Exception)
    async def unhandled(request: Request, exc: Exception) -> JSONResponse:
        log.exception("unhandled error")
        return SafeJSONResponse({"error": {"message": "Internal error. Quote the request id when reporting it."},
                             "request_id": getattr(request.state, "request_id", None)}, status_code=500)

    def openapi() -> dict[str, Any]:
        """OpenAPI with the served model's generated `Applicant` schema (regenerated when the model changes)."""
        current: LoadedModel | None = holder.current
        key = current.version if current else None
        if app.openapi_schema and app.state.openapi_for == key:
            return app.openapi_schema
        spec = get_openapi(title=app.title, version=app.version, description=app.description, routes=app.routes,
                           tags=app.openapi_tags)
        comps = spec.setdefault("components", {}).setdefault("schemas", {})
        if current is not None:
            applicant = current.request_model.model_json_schema(ref_template="#/components/schemas/{model}")
            applicant["description"] = (f"Inputs of {current.name} v{current.version}. Every field is required; null "
                                        "means unknown (imputed).")
            comps["Applicant"] = applicant
        else:
            comps["Applicant"] = {"type": "object", "description": "No model loaded"}
        for name in ("ScoreRequest",):
            props = comps.get(name, {}).get("properties", {})
            if "applicant" in props:
                props["applicant"] = {"$ref": "#/components/schemas/Applicant"}
        comps.setdefault("BatchScoreRequest", BatchScoreRequest.model_json_schema())
        batch = comps["BatchScoreRequest"].get("properties", {})
        if "applicants" in batch:
            batch["applicants"] = {"type": "array", "minItems": 1, "items": {"$ref": "#/components/schemas/Applicant"}}
        spec.setdefault("components", {})["securitySchemes"] = {
            "APIKeyHeader": {"type": "apiKey", "in": "header", "name": "X-API-Key"}}
        app.openapi_schema = spec
        app.state.openapi_for = key
        return spec

    app.state.openapi_for = None
    app.openapi = openapi  # type: ignore[method-assign]
    return app


def run() -> None:
    """``creditiq-api``: serve with uvicorn (HOST, PORT, WORKERS from the environment)."""
    import uvicorn

    uvicorn.run("creditiq_api.main:create_app", factory=True, host=os.environ.get("HOST", "0.0.0.0"),
                port=int(os.environ.get("PORT", "8000")), workers=int(os.environ.get("WORKERS", "1")),
                log_config=None, access_log=False)
