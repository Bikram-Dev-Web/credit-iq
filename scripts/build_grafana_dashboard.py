"""Generate infra/grafana/dashboards/creditiq-overview.json (API latency, throughput, errors, PD distribution,
flag rate, model version, drift). Edit this script, not the JSON.

Usage:  python scripts/build_grafana_dashboard.py
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

OUT = Path(__file__).resolve().parent.parent / "infra" / "grafana" / "dashboards" / "creditiq-overview.json"
DS = {"type": "prometheus", "uid": "prometheus"}
_id = 0


def panel(title: str, kind: str, targets: list[tuple[str, str]], x: int, y: int, w: int, h: int,
          unit: str = "short", extra: dict[str, Any] | None = None, desc: str = "") -> dict[str, Any]:
    global _id
    _id += 1
    p: dict[str, Any] = {
        "id": _id, "title": title, "type": kind, "datasource": DS, "description": desc,
        "gridPos": {"x": x, "y": y, "w": w, "h": h},
        "targets": [{"refId": chr(65 + i), "datasource": DS, "expr": e, "legendFormat": leg} for i, (e, leg) in
                    enumerate(targets)],
        "fieldConfig": {"defaults": {"unit": unit, "color": {"mode": "palette-classic"}}, "overrides": []},
        "options": {},
    }
    if kind == "timeseries":
        p["fieldConfig"]["defaults"]["custom"] = {"lineWidth": 2, "fillOpacity": 8, "showPoints": "never"}
        p["options"] = {"legend": {"displayMode": "list", "placement": "bottom"}, "tooltip": {"mode": "multi"}}
    if extra:
        for k, v in extra.items():
            if k == "defaults":
                p["fieldConfig"]["defaults"].update(v)
            else:
                p[k] = v
    return p


SCORE = 'route=~"/v1/score|/v1/decide"'

panels = [
    panel("Model version", "stat", [("max(creditiq_model_version)", "")], 0, 0, 4, 4,
          extra={"options": {"colorMode": "value", "graphMode": "none", "reduceOptions": {"calcs": ["lastNotNull"]}}}),
    panel("Requests / s", "stat", [("sum(rate(creditiq_http_requests_total[1m]))", "")], 4, 0, 4, 4, "reqps",
          extra={"options": {"graphMode": "area", "reduceOptions": {"calcs": ["lastNotNull"]}}}),
    panel("p95 score latency", "stat",
          [(f"histogram_quantile(0.95, sum by (le) (rate(creditiq_http_request_duration_seconds_bucket{{{SCORE}}}[5m])))", "")],
          8, 0, 4, 4, "s", extra={"defaults": {"thresholds": {"mode": "absolute", "steps": [
              {"color": "green", "value": None}, {"color": "orange", "value": 0.05}, {"color": "red", "value": 0.1}]}},
              "options": {"colorMode": "value", "reduceOptions": {"calcs": ["lastNotNull"]}}}),
    panel("Error rate", "stat", [("sum(rate(creditiq_http_errors_total[5m]))", "")], 12, 0, 4, 4, "reqps",
          extra={"defaults": {"thresholds": {"mode": "absolute", "steps": [{"color": "green", "value": None},
                                                                          {"color": "red", "value": 0.01}]}},
                 "options": {"colorMode": "value", "reduceOptions": {"calcs": ["lastNotNull"]}}}),
    panel("Flag rate (1 h)", "stat",
          [('sum(rate(creditiq_decisions_total{outcome="flagged_for_review"}[1h])) / sum(rate(creditiq_decisions_total[1h]))', "")],
          16, 0, 4, 4, "percentunit", extra={"options": {"reduceOptions": {"calcs": ["lastNotNull"]}}}),
    panel("Drift status", "stat", [("max(creditiq_drift_status)", "")], 20, 0, 4, 4,
          extra={"defaults": {"mappings": [{"type": "value", "options": {
              "0": {"text": "Stable", "color": "green"}, "1": {"text": "Warning", "color": "orange"},
              "2": {"text": "Critical", "color": "red"}}}]},
              "options": {"colorMode": "background", "reduceOptions": {"calcs": ["lastNotNull"]}}},
          desc="From the latest monitoring run (monitor job or /v1/monitoring/check)"),
    panel("Latency by percentile (score + decide)", "timeseries", [
        (f"histogram_quantile(0.50, sum by (le) (rate(creditiq_http_request_duration_seconds_bucket{{{SCORE}}}[5m])))", "p50"),
        (f"histogram_quantile(0.95, sum by (le) (rate(creditiq_http_request_duration_seconds_bucket{{{SCORE}}}[5m])))", "p95"),
        (f"histogram_quantile(0.99, sum by (le) (rate(creditiq_http_request_duration_seconds_bucket{{{SCORE}}}[5m])))", "p99"),
    ], 0, 4, 12, 8, "s"),
    panel("Throughput by route", "timeseries", [("sum by (route) (rate(creditiq_http_requests_total[1m]))", "{{route}}")],
          12, 4, 12, 8, "reqps"),
    panel("Responses by status", "timeseries", [("sum by (status) (rate(creditiq_http_requests_total[1m]))", "{{status}}")],
          0, 12, 8, 7, "reqps"),
    panel("Errors by route", "timeseries", [("sum by (route, type) (rate(creditiq_http_errors_total[5m]))", "{{route}} {{type}}")],
          8, 12, 8, 7, "reqps"),
    panel("Audit-log write p95", "timeseries",
          [("histogram_quantile(0.95, sum by (le) (rate(creditiq_audit_write_seconds_bucket[5m])))", "p95")], 16, 12, 8, 7, "s"),
    panel("PD distribution of scored applicants", "bargauge",
          [("sum by (le) (increase(creditiq_pd_bucket[1h]))", "≤ {{le}}")], 0, 19, 12, 8,
          extra={"options": {"displayMode": "gradient", "orientation": "vertical", "reduceOptions": {"calcs": ["lastNotNull"]}}},
          desc="Cumulative histogram buckets over the last hour"),
    panel("Decisions by outcome", "timeseries",
          [("sum by (outcome) (rate(creditiq_decisions_total[5m]))", "{{outcome}}")], 12, 19, 12, 8, "reqps"),
    panel("PSI by input (latest run)", "bargauge", [("sort_desc(creditiq_drift_feature_psi)", "{{feature}}")], 0, 27, 12, 10,
          extra={"defaults": {"min": 0, "max": 0.5, "thresholds": {"mode": "absolute", "steps": [
              {"color": "green", "value": None}, {"color": "orange", "value": 0.1}, {"color": "red", "value": 0.25}]}},
              "options": {"displayMode": "lcd", "orientation": "horizontal", "reduceOptions": {"calcs": ["lastNotNull"]}}}),
    panel("PD distribution PSI", "timeseries", [("max(creditiq_drift_score_psi)", "score PSI")], 12, 27, 12, 5,
          extra={"defaults": {"thresholds": {"mode": "absolute", "steps": [{"color": "green", "value": None},
                                                                          {"color": "orange", "value": 0.1}]}}}),
    panel("Drift alerts (latest run)", "timeseries", [("sum by (severity) (creditiq_drift_alerts)", "{{severity}}")],
          12, 32, 12, 5),
    panel("Rules fired", "timeseries", [("sum by (rule) (rate(creditiq_policy_rules_fired_total[5m]))", "{{rule}}")],
          0, 37, 24, 6, "reqps"),
]

dashboard = {
    "uid": "creditiq-overview", "title": "CreditIQ — API, decisions and drift", "tags": ["creditiq"],
    "timezone": "browser", "schemaVersion": 39, "version": 1, "refresh": "30s",
    "time": {"from": "now-6h", "to": "now"}, "panels": panels,
    "description": "Educational decision-support prototype, not for real credit decisions.",
}

if __name__ == "__main__":
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(dashboard, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {OUT}")
