"""The scheduled drift job.

Two sources, both compared with the served model's training reference:

* **Inbox batches** — CSV files of new applications dropped into ``MONITOR_INBOX`` (feature PSI, missingness,
  schema violations, PD drift and, when labels are present, re-measured performance). Each file is processed once
  and moved to ``processed/``.
* **The audit log** — PDs of recent decisions (the log holds scores, never inputs), giving PD drift without
  storing any applicant data.

Results go to Postgres (``monitoring_runs`` and ``alerts``); the API exposes them to Prometheus. Alerts are
logged and, if ``ALERT_WEBHOOK_URL`` is set, posted as JSON.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import urllib.request
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
from sqlalchemy import select

from creditiq_api import drift
from creditiq_api.db import Alert, Database, Decision, MonitoringRun
from creditiq_api.main import make_registry
from creditiq_api.model_store import LoadedModel, load_package
from creditiq_api.settings import Settings
from creditiq_core import monitoring as mon
from creditiq_core.data.csv import read_csv_file

log = logging.getLogger("creditiq.monitor")


def thresholds_from_env() -> mon.DriftThresholds:
    return mon.DriftThresholds(
        psi_moderate=float(os.environ.get("DRIFT_PSI_MODERATE", 0.10)),
        psi_significant=float(os.environ.get("DRIFT_PSI_SIGNIFICANT", 0.25)),
        score_psi=float(os.environ.get("DRIFT_SCORE_PSI", 0.10)),
        missing_change=float(os.environ.get("DRIFT_MISSING_CHANGE", 0.10)),
    )


def served_model(settings: Settings) -> LoadedModel:
    registry = make_registry(settings)
    version = registry.resolve(settings.model_alias)
    if version is None:
        raise RuntimeError(f"No version of {settings.model_name} has the '{settings.model_alias}' alias.")
    return load_package(registry.model_name, version, registry.fetch(version))


def notify(run: dict[str, Any]) -> None:
    if not run.get("alerts"):
        return
    for a in run["alerts"]:
        log.warning("drift alert", extra={"code": a["code"], "severity": a["severity"], "alert": a["message"],
                                          "run_id": run["id"], "model_version": run["model_version"]})
    url = os.environ.get("ALERT_WEBHOOK_URL")
    if not url:
        return
    body = json.dumps({"text": f"CreditIQ drift: {run['status']} on {run['batch_name']} "
                               f"(model v{run['model_version']}): " + "; ".join(a["message"] for a in run["alerts"]),
                       "run": {k: run[k] for k in ("id", "status", "batch_name", "model_version")},
                       "alerts": run["alerts"]}, default=str).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
    try:
        urllib.request.urlopen(req, timeout=10).close()
    except OSError as exc:
        log.error("alert webhook failed", extra={"error": str(exc)})


def check_file(model: LoadedModel, db: Database, path: Path, source: str,
               thresholds: mon.DriftThresholds) -> dict[str, Any]:
    frame, _info = read_csv_file(path.read_bytes(), name=path.name, optimize_memory=False)
    run = drift.run_check(model, frame, path.name, source, db, thresholds)
    notify(run)
    return run


def process_inbox(model: LoadedModel, db: Database, inbox: Path, thresholds: mon.DriftThresholds) -> list[dict[str, Any]]:
    inbox.mkdir(parents=True, exist_ok=True)
    done = inbox / "processed"
    done.mkdir(exist_ok=True)
    runs = []
    for path in sorted(inbox.glob("*.csv")):
        try:
            runs.append(check_file(model, db, path, "scheduled", thresholds))
            target = done / f"{datetime.now(UTC):%Y%m%dT%H%M%S}_{path.name}"
            shutil.move(str(path), target)
        except Exception:
            log.exception("inbox file failed", extra={"file": path.name})
    return runs


def audit_score_drift(model: LoadedModel, db: Database, hours: float, thresholds: mon.DriftThresholds,
                      min_rows: int = 200) -> dict[str, Any] | None:
    """PD drift of recent live decisions (from the audit log) against the validation PD reference."""
    since = datetime.now(UTC) - timedelta(hours=hours)
    with db.session() as s:
        pds = list(s.scalars(select(Decision.pd).where(Decision.created_at >= since,
                                                       Decision.model_version == model.version)))
    if len(pds) < min_rows:
        log.info("audit-log PD drift skipped", extra={"rows": len(pds), "min_rows": min_rows})
        return None
    score = mon.score_drift(model.artifact.reference, np.asarray(pds), thresholds)
    alerts = []
    if score and score["psi"] >= thresholds.score_psi:
        alerts.append(mon.DriftAlert("score_shift", "warning",
                                     f"Live PDs from the audit log have shifted (PSI = {score['psi']:.3f}).",
                                     float(score["psi"]), thresholds.score_psi))
    with db.session() as s:
        run = MonitoringRun(source="audit-log", model_name=model.name, model_version=model.version,
                            batch_name=f"decisions, last {hours:g} h", n_rows=len(pds),
                            status="warning" if alerts else "ok", max_feature_psi=None, n_significant=0,
                            score_psi=score["psi"] if score else None, mean_pd=float(np.mean(pds)),
                            ref_mean_pd=score["ref_mean_pd"] if score else None, labels_available=False,
                            observed_default_rate=None, roc_auc=None, features=[], score=score, schema_check={},
                            thresholds={"score_psi": thresholds.score_psi})
        s.add(run)
        s.flush()
        for a in alerts:
            s.add(Alert(run_id=run.id, code=a.code, severity=a.severity, message=a.message, value=a.value,
                        threshold=a.threshold, columns=[]))
        run_id = run.id
    out = drift.run_to_dict(drift.fetch_run(db, run_id))
    notify(out)
    return out
