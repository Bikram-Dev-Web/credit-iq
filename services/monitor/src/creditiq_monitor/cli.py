"""``creditiq-monitor``: run the drift job once, on a schedule, or on one file.

    creditiq-monitor run                         # inbox files + audit-log PD drift, once
    creditiq-monitor schedule --every 3600       # loop forever (the Docker service)
    creditiq-monitor check data/sample_monitoring_batch.csv --source replay
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

from creditiq_api import migrate
from creditiq_api.db import Database
from creditiq_api.model_store import LoadedModel
from creditiq_api.observability import configure_logging
from creditiq_api.settings import get_settings
from creditiq_monitor import job

log = logging.getLogger("creditiq.monitor")


def _setup() -> tuple[Database, LoadedModel]:
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_json)
    if settings.auto_migrate:
        migrate.upgrade(settings.database_url)
    return Database(settings.database_url), job.served_model(settings)


def run_once(inbox: Path, hours: float) -> int:
    db, model = _setup()
    t = job.thresholds_from_env()
    runs = job.process_inbox(model, db, inbox, t)
    audit = job.audit_score_drift(model, db, hours, t)
    runs += [audit] if audit else []
    worst = max((r["status"] for r in runs), key=["ok", "warning", "critical"].index, default="ok")
    log.info("monitoring run finished", extra={"runs": len(runs), "worst_status": worst,
                                               "model_version": model.version})
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="creditiq-monitor", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("run", "schedule"):
        p = sub.add_parser(name)
        p.add_argument("--inbox", type=Path, default=Path(os.environ.get("MONITOR_INBOX", "storage/inbox")))
        p.add_argument("--hours", type=float, default=float(os.environ.get("MONITOR_AUDIT_HOURS", 24)),
                       help="Audit-log window for PD drift")
        if name == "schedule":
            p.add_argument("--every", type=float, default=float(os.environ.get("MONITOR_EVERY_SECONDS", 3600)))
    c = sub.add_parser("check", help="Check one CSV now")
    c.add_argument("path", type=Path)
    c.add_argument("--source", default="replay")
    args = ap.parse_args(argv)

    if args.cmd == "run":
        return run_once(args.inbox, args.hours)
    if args.cmd == "schedule":
        while True:
            try:
                run_once(args.inbox, args.hours)
            except Exception:
                log.exception("monitoring run failed")
            time.sleep(args.every)
    db, model = _setup()
    run = job.check_file(model, db, args.path, args.source, job.thresholds_from_env())
    print(json.dumps({k: run[k] for k in ("id", "status", "batch_name", "model_version", "n_rows", "score_psi",
                                          "max_feature_psi", "roc_auc")} | {"alerts": [a["message"] for a in
                                                                                       run["alerts"]]},
                     indent=2, default=str), file=sys.stdout)
    return 0 if run["status"] == "ok" else 3


if __name__ == "__main__":
    raise SystemExit(main())
