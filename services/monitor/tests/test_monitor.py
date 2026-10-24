"""The drift job: inbox batches, audit-log PD drift, alert webhook."""
from __future__ import annotations

import json
import shutil
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from sqlalchemy import select

from creditiq_api import migrate
from creditiq_api.db import Alert, Database, Decision, MonitoringRun
from creditiq_api.settings import Settings
from creditiq_monitor import cli, job
from creditiq_trainer import file_registry, pipeline
from creditiq_trainer.config import load_config
from creditiq_trainer.package_writer import write_package

ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture(scope="module")
def env(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    tmp = tmp_path_factory.mktemp("monitor")
    cfg = load_config(ROOT / "services/trainer/configs/synthetic_ci.yaml").model_copy(update={"sample_size": 3000})
    cfg.dataset.path = str(ROOT / "data/sample_credit_data.csv")
    data = pipeline.prepare(cfg)
    pkg = write_package(cfg, data, pipeline.train(cfg, data), tmp / "pkg")
    file_registry.add_version(tmp / "reg", "creditiq-pd", pkg, version="1", aliases=["production"])
    url = f"sqlite:///{(tmp / 'm.db').as_posix()}"
    migrate.upgrade(url)
    settings = Settings(_env_file=None, environment="test", database_url=url, file_registry_path=str(tmp / "reg"),  # type: ignore[call-arg]
                        audit_hash_key="k" * 20)
    return {"tmp": tmp, "db": Database(url), "model": job.served_model(settings), "url": url, "reg": tmp / "reg"}


def test_inbox_batch_raises_drift_alerts_and_is_moved(env: dict[str, Any]) -> None:
    inbox = env["tmp"] / "inbox"
    inbox.mkdir(exist_ok=True)
    shutil.copy(ROOT / "data/sample_monitoring_batch.csv", inbox / "batch.csv")
    runs = job.process_inbox(env["model"], env["db"], inbox, job.thresholds_from_env())
    assert len(runs) == 1 and runs[0]["status"] in ("warning", "critical")
    assert {a["code"] for a in runs[0]["alerts"]} & {"missingness_jump", "score_shift", "feature_shift"}
    assert runs[0]["labels_available"] and runs[0]["roc_auc"] is not None
    assert not (inbox / "batch.csv").exists() and len(list((inbox / "processed").glob("*.csv"))) == 1
    with env["db"].session() as s:
        assert s.scalars(select(Alert)).first() is not None


def test_same_population_is_stable(env: dict[str, Any], tmp_path: Path) -> None:
    import pandas as pd

    sample = pd.read_csv(ROOT / "data/sample_credit_data.csv").sample(4000, random_state=3)
    path = tmp_path / "same.csv"
    sample.to_csv(path, index=False)
    run = job.check_file(env["model"], env["db"], path, "replay", job.thresholds_from_env())
    assert run["status"] == "ok", run["alerts"]


def test_audit_log_pd_drift(env: dict[str, Any]) -> None:
    db, model = env["db"], env["model"]
    rng = np.random.default_rng(0)
    with db.session() as s:
        for i, p in enumerate(rng.uniform(0.3, 0.9, 300)):  # live PDs far above the validation reference
            s.add(Decision(request_id=f"r{i}", endpoint="score", api_key_id="t", role="analyst",
                           model_name=model.name, model_version=model.version, policy_version="p", pd=float(p),
                           raw_score=float(p), band="High", outcome="flagged_for_review", threshold=0.2,
                           threshold_source="model", rules_fired=[], top_reasons=[], ead=1.0, expected_loss=1.0,
                           input_hash="x" * 64, hash_key_id="k1"))
    run = job.audit_score_drift(model, db, hours=24, thresholds=job.thresholds_from_env())
    assert run is not None and run["source"] == "audit-log" and run["status"] == "warning"
    assert run["alerts"][0]["code"] == "score_shift"
    assert job.audit_score_drift(model, db, hours=24, thresholds=job.thresholds_from_env(), min_rows=10_000) is None


def test_webhook_receives_alerts(env: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    received: list[dict[str, Any]] = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            received.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            self.send_response(204)
            self.end_headers()

        def log_message(self, *a: Any) -> None:
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.handle_request, daemon=True).start()
    monkeypatch.setenv("ALERT_WEBHOOK_URL", f"http://127.0.0.1:{server.server_port}/hook")
    job.check_file(env["model"], env["db"], ROOT / "data/sample_monitoring_batch.csv", "replay",
                   job.thresholds_from_env())
    server.server_close()
    assert received and received[0]["alerts"] and "CreditIQ drift" in received[0]["text"]


def test_cli_check_exit_code(env: dict[str, Any], monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    from creditiq_api.settings import get_settings

    get_settings.cache_clear()
    for k, v in {"DATABASE_URL": env["url"], "FILE_REGISTRY_PATH": str(env["reg"]), "ENVIRONMENT": "test",
                 "AUDIT_HASH_KEY": "k" * 20}.items():
        monkeypatch.setenv(k, v)
    code = cli.main(["check", str(ROOT / "data/sample_monitoring_batch.csv")])
    get_settings.cache_clear()
    out = capsys.readouterr().out  # JSON log lines, then the indented result document
    result = json.loads(out[out.index("{\n"):])
    assert code == 3 and result["alerts"] and result["status"] != "ok"
    with env["db"].session() as s:
        assert s.scalars(select(MonitoringRun).where(MonitoringRun.source == "replay")).first() is not None
