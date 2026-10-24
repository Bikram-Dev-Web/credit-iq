"""Structured JSON logging with request ids, and Prometheus metrics."""
from __future__ import annotations

import contextlib
import contextvars
import json
import logging
import sys
import time
import uuid
from collections.abc import Awaitable, Callable, Iterable
from datetime import UTC, datetime
from typing import Any

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram
from prometheus_client.core import GaugeMetricFamily
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
from starlette.routing import Match

request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="-")

# ---------------------------------------------------------------------------- logging

_RESERVED = set(vars(logging.LogRecord("", 0, "", 0, "", None, None))) | {"message", "asctime"}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {"ts": datetime.fromtimestamp(record.created, UTC).isoformat(timespec="milliseconds"),
                                   "level": record.levelname.lower(), "logger": record.name,
                                   "msg": record.getMessage(), "request_id": request_id_var.get()}
        for k, v in vars(record).items():
            if k not in _RESERVED and not k.startswith("_"):
                payload[k] = v
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging(level: str = "INFO", json_logs: bool = True) -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter() if json_logs else logging.Formatter("%(levelname)s %(name)s %(message)s"))
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level.upper())
    for noisy in ("uvicorn.access", "alembic", "mlflow"):
        logging.getLogger(noisy).setLevel("WARNING")


def new_request_id() -> str:
    return uuid.uuid4().hex


# ---------------------------------------------------------------------------- metrics

LATENCY_BUCKETS = (0.002, 0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0)
PD_BUCKETS = (0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.5, 0.75, 1.0)


class Metrics:
    """All application metrics, on a dedicated registry (one per app instance, which keeps tests isolated)."""

    def __init__(self) -> None:
        self.registry = CollectorRegistry()
        r = self.registry
        self.requests = Counter("creditiq_http_requests_total", "HTTP requests", ["method", "route", "status"],
                                registry=r)
        self.latency = Histogram("creditiq_http_request_duration_seconds", "HTTP request latency",
                                 ["method", "route"], buckets=LATENCY_BUCKETS, registry=r)
        self.errors = Counter("creditiq_http_errors_total", "Requests that ended in a 5xx or an exception",
                              ["route", "type"], registry=r)
        self.pd = Histogram("creditiq_pd", "Predicted probability of default of scored applicants", ["endpoint"],
                            buckets=PD_BUCKETS, registry=r)
        self.decisions = Counter("creditiq_decisions_total", "Scored applicants by outcome", ["endpoint", "outcome"],
                                 registry=r)
        self.rules_fired = Counter("creditiq_policy_rules_fired_total", "Hard-rule hits", ["rule"], registry=r)
        self.model_info = Gauge("creditiq_model_info", "Served model version (value 1)", ["model", "version"],
                                registry=r)
        self.model_version = Gauge("creditiq_model_version", "Served model version number", registry=r)
        self.model_reloads = Counter("creditiq_model_reloads_total", "Model versions loaded", registry=r)
        self.policy_info = Gauge("creditiq_policy_info", "Active policy version (value 1)", ["version"], registry=r)
        self.audit_seconds = Histogram("creditiq_audit_write_seconds", "Audit-log write time",
                                       buckets=(0.0005, 0.001, 0.002, 0.005, 0.01, 0.025, 0.05, 0.1), registry=r)
        self.batch_rows = Counter("creditiq_batch_rows_total", "Rows scored through the batch endpoint", registry=r)

    def set_model(self, name: str, version: str) -> None:
        self.model_info.clear()
        self.model_info.labels(name, version).set(1)
        with contextlib.suppress(ValueError):
            self.model_version.set(float(version))
        self.model_reloads.inc()

    def set_policy(self, version: str) -> None:
        self.policy_info.clear()
        self.policy_info.labels(version).set(1)


class DriftCollector:
    """Exposes the latest monitoring run (written by the monitor job) as gauges, read from the database on scrape."""

    def __init__(self, fetch: Callable[[], dict[str, Any] | None]):
        self.fetch = fetch

    def collect(self) -> Iterable[GaugeMetricFamily]:
        run = self.fetch()
        if not run:
            return []
        feat = GaugeMetricFamily("creditiq_drift_feature_psi", "PSI per model input (latest monitoring run)",
                                 labels=["feature"])
        for row in run.get("features", []):
            if row.get("psi") is not None:
                feat.add_metric([str(row["column"])], float(row["psi"]))
        score = GaugeMetricFamily("creditiq_drift_score_psi", "PSI of the predicted-PD distribution (latest run)")
        if run.get("score_psi") is not None:
            score.add_metric([], float(run["score_psi"]))
        alerts = GaugeMetricFamily("creditiq_drift_alerts", "Alerts raised by the latest monitoring run",
                                   labels=["severity"])
        for sev in ("warning", "critical"):
            alerts.add_metric([sev], float(sum(1 for a in run.get("alerts", []) if a.get("severity") == sev)))
        status = GaugeMetricFamily("creditiq_drift_status", "Latest run status (0 ok, 1 warning, 2 critical)")
        status.add_metric([], float({"ok": 0, "warning": 1, "critical": 2}.get(run.get("status", "ok"), 0)))
        age = GaugeMetricFamily("creditiq_drift_last_run_timestamp_seconds", "When the latest monitoring run happened")
        age.add_metric([], float(run["created_at"].timestamp()))
        return [feat, score, alerts, status, age]


def route_template(request: Request) -> str:
    for route in request.app.routes:
        match, _ = route.matches(request.scope)
        if match == Match.FULL:
            return str(getattr(route, "path", request.url.path))
    return "unmatched"


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Request id in/out (X-Request-ID), access log line, latency and error metrics."""

    async def dispatch(self, request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
        rid = request.headers.get("x-request-id") or new_request_id()
        if len(rid) > 64 or not rid.replace("-", "").isalnum():
            rid = new_request_id()
        token = request_id_var.set(rid)
        request.state.request_id = rid
        metrics: Metrics = request.app.state.metrics
        route = route_template(request)
        t0 = time.perf_counter()
        status_code = 500
        try:
            response = await call_next(request)
            status_code = response.status_code
        except Exception as exc:
            metrics.errors.labels(route, type(exc).__name__).inc()
            raise
        finally:
            elapsed = time.perf_counter() - t0
            if route != "/metrics":
                metrics.requests.labels(request.method, route, str(status_code)).inc()
                metrics.latency.labels(request.method, route).observe(elapsed)
                if status_code >= 500:
                    metrics.errors.labels(route, f"http_{status_code}").inc()
                logging.getLogger("creditiq.access").info(
                    "request", extra={"method": request.method, "route": route, "status": status_code,
                                      "duration_ms": round(elapsed * 1000, 2)})
            request_id_var.reset(token)
        response.headers["X-Request-ID"] = rid
        return response
