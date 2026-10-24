"""API behaviour: scoring, validation, auth, batch, models, policies, analytics, monitoring, audit, metrics."""
from __future__ import annotations

import io
import json
import logging
import threading
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from creditiq_api.db import AdminEvent, Decision

ROOT = Path(__file__).resolve().parents[3]
A = {"X-API-Key": "test-analyst-key-123"}
ADM = {"X-API-Key": "test-admin-key-456"}

NUMERIC_STRUCTURAL = {"row", "limit", "offset", "version", "importance_rank", "band_n", "min_group_size", "created",
                      "bin", "risk_group", "rows", "status_code"}


def numeric_keys(obj: Any, key: str | None = None) -> set[str]:
    """Names of every numeric leaf in a JSON document."""
    out: set[str] = set()
    if isinstance(obj, bool) or obj is None:
        return out
    if isinstance(obj, int | float):
        return {key} if key else set()
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in ("provenance", "thresholds", "assumptions", "config", "params", "lineage", "json_schema",
                     "split_sizes", "schema_check", "threshold_sources", "rules", "summary") and isinstance(v, dict):
                if k in ("assumptions", "summary"):
                    out |= numeric_keys(v, None) if k == "summary" else set()
                continue
            out |= numeric_keys(v, k)
    if isinstance(obj, list):
        for v in obj:
            out |= numeric_keys(v, key)
    return out


def assert_provenance(body: dict[str, Any]) -> None:
    """Contract: every number in a response is labelled in its provenance map."""
    assert body["data_provenance"] in ("synthetic", "observed")
    prov = body["provenance"]
    labels = {"observed", "measured", "model", "assumption", "simulated", "synthetic"}
    for k, v in prov.items():
        assert v and set(v) <= labels, (k, v)
    missing = numeric_keys(body) - set(prov) - NUMERIC_STRUCTURAL
    assert not missing, f"numbers without provenance: {sorted(missing)}"


# ---------------------------------------------------------------- health, meta, auth
def test_health_ready_and_meta(client: TestClient):
    assert client.get("/healthz").json() == {"status": "ok"}
    r = client.get("/readyz")
    assert r.status_code == 200 and r.json()["checks"] == {"model": True, "database": True, "policy": True}
    m = client.get("/v1/meta", headers=A).json()
    assert m["model_version"] == "1" and m["policy_version"] == "2026.10.1" and m["role"] == "analyst"
    assert m["data_provenance"] == "synthetic" and "not for real credit decisions" in m["disclaimer"]
    assert r.headers["X-Request-ID"]


def test_auth_roles(client: TestClient, app_factory):
    assert client.get("/v1/models").status_code == 401
    assert client.get("/v1/models", headers={"X-API-Key": "wrong-key-123"}).status_code == 401
    assert client.get("/v1/models", headers={"Authorization": "Bearer test-analyst-key-123"}).status_code == 200
    r = client.post("/v1/models/2/promote", headers=A)
    assert r.status_code == 403 and "admin" in r.json()["error"]["message"]
    locked = app_factory(allow_admin_actions=False)
    r = locked.post("/v1/models/2/promote", headers=ADM)
    assert r.status_code == 403 and "disabled" in r.json()["error"]["message"]


# ---------------------------------------------------------------- scoring
def test_score_returns_the_full_contract(client: TestClient, applicant):
    r = client.post("/v1/score", headers=A, json={"applicant": applicant})
    assert r.status_code == 200, r.text
    b = r.json()
    assert 0 <= b["pd"] <= 1 and b["model_version"] == "1" and b["policy_version"] == "2026.10.1"
    assert b["outcome"] in ("within_threshold", "flagged_for_review")
    assert b["flagged_for_review"] == (b["outcome"] == "flagged_for_review") == (b["pd"] >= b["threshold"])
    assert b["expected_loss"] == pytest.approx(b["pd"] * b["lgd"] * b["ead"])
    assert b["risk_band"] in ("Low", "Moderate", "Elevated", "High")
    assert 1 <= len(b["top_reasons"]) <= 8 and all(set(x) >= {"feature", "contribution", "direction"}
                                                      for x in b["top_reasons"])
    assert {"CODE_GENDER", "DAYS_BIRTH"}.isdisjoint(x["feature"] for x in b["top_reasons"])
    assert b["uncertainty"]["test_roc_auc"] > 0.6 and b["request_id"] == r.headers["X-Request-ID"]
    assert any("SYNTHETIC" in c for c in b["caveats"])
    assert "approve" not in r.text.lower() and "deny" not in r.text.lower()
    assert_provenance(b)


def test_validation_errors_are_clear_422s(client: TestClient, applicant):
    missing = dict(applicant)
    missing.pop("EXT_SOURCE_2")
    r = client.post("/v1/score", headers=A, json={"applicant": missing})
    assert r.status_code == 422
    err = r.json()["error"]["errors"][0]
    assert err["loc"] == ["body", "applicant", "EXT_SOURCE_2"] and "null" in err["msg"]
    r = client.post("/v1/score", headers=A, json={"applicant": {**applicant, "CODE_GENDER": "F"}})
    assert r.status_code == 422 and r.json()["error"]["errors"][0]["loc"][-1] == "CODE_GENDER"  # sensitive: rejected
    r = client.post("/v1/score", headers=A, json={"applicant": {**applicant, "AMT_CREDIT": "lots"}})
    assert r.status_code == 422
    r = client.post("/v1/score", headers=A, json={"applicant": applicant, "surprise": 1})
    assert r.status_code == 422
    nulls = {k: None for k in applicant}
    r = client.post("/v1/score", headers=A, json={"applicant": nulls})
    assert r.status_code == 200  # every value unknown: imputed from training statistics


def test_input_warnings_for_out_of_range_and_unseen_values(client: TestClient, applicant):
    weird = {**applicant, "AMT_INCOME_TOTAL": 1e12, "NAME_CONTRACT_TYPE": "Crypto loans"}
    b = client.post("/v1/score", headers=A, json={"applicant": weird}).json()
    assert any("AMT_INCOME_TOTAL" in w and "outside" in w for w in b["warnings"])
    assert any("Crypto loans" in w for w in b["warnings"])


def test_decide_reports_rules(client: TestClient, applicant):
    stretched = {**applicant, "AMT_ANNUITY": 900_000.0, "AMT_INCOME_TOTAL": 1_000_000.0}
    b = client.post("/v1/decide", headers=A, json={"applicant": stretched}).json()
    assert "annuity_to_income" in b["rules_fired"] and b["outcome"] == "flagged_for_review"
    rule = next(r for r in b["rules"] if r["id"] == "annuity_to_income")
    assert rule["value"] == pytest.approx(0.9) and rule["limit"] == 0.45
    assert any("annuity_to_income" in x for x in b["reasons"]) and b["policy_name"]
    assert_provenance(b)


def test_audit_log_holds_no_raw_inputs(client: TestClient, applicant, caplog):
    marker = 987654.321
    caplog.set_level(logging.DEBUG)
    r = client.post("/v1/decide", headers=A, json={"applicant": {**applicant, "AMT_GOODS_PRICE": marker}})
    rid = r.json()["request_id"]
    with client.app.state.db.session() as s:
        row = s.scalar(select(Decision).where(Decision.request_id == rid))
        dumped = json.dumps({c.name: str(getattr(row, c.name)) for c in Decision.__table__.columns})
    assert row is not None and row.endpoint == "decide" and len(row.input_hash) == 64
    assert "987654" not in dumped and "AMT_GOODS_PRICE\": 98" not in dumped
    assert "987654" not in caplog.text  # the request body is never logged
    # the same inputs give the same HMAC (matchable), a different key gives a different one
    r2 = client.post("/v1/decide", headers=A, json={"applicant": {**applicant, "AMT_GOODS_PRICE": marker}})
    with client.app.state.db.session() as s:
        row2 = s.scalar(select(Decision).where(Decision.request_id == r2.json()["request_id"]))
    assert row2.input_hash == row.input_hash


def test_audit_store_down_means_no_decision(client: TestClient, applicant, monkeypatch):
    from sqlalchemy.exc import OperationalError

    def broken(*a, **k):
        raise OperationalError("insert", {}, Exception("database is down"))

    monkeypatch.setattr(client.app.state.db, "session", broken)
    r = client.post("/v1/score", headers=A, json={"applicant": applicant})
    assert r.status_code == 503 and "audit" in r.json()["error"]["message"]


def test_rate_limit(app_factory, applicant):
    c = app_factory(rate_limit_score="2/minute")
    codes = [c.post("/v1/score", headers=A, json={"applicant": applicant}).status_code for _ in range(3)]
    assert codes == [200, 200, 429]


# ---------------------------------------------------------------- batch
def test_batch_json_and_csv(client: TestClient, applicant):
    rows = [applicant, {**applicant, "AMT_ANNUITY": 900_000.0, "AMT_INCOME_TOTAL": 1_000_000.0}]
    r = client.post("/v1/score/batch", headers=A, json={"applicants": rows})
    assert r.status_code == 200 and r.headers["content-type"].startswith("application/x-ndjson")
    lines = [json.loads(x) for x in r.text.strip().splitlines()]
    assert lines[0]["meta"]["rows"] == 2 and len(lines) == 3
    assert lines[2]["outcome"] == "flagged_for_review" and "annuity_to_income" in lines[2]["rules_fired"]
    csv_body = pd.DataFrame(rows).to_csv(index=False)
    r = client.post("/v1/score/batch?format=csv", headers={**A, "Content-Type": "text/csv"}, content=csv_body)
    assert r.status_code == 200, r.text
    out = pd.read_csv(io.StringIO(r.text))
    assert len(out) == 2 and set(out["outcome"]) <= {"within_threshold", "flagged_for_review"}
    assert out["pd"].iloc[0] == pytest.approx(lines[1]["pd"])
    with client.app.state.db.session() as s:
        n = len(s.scalars(select(Decision).where(Decision.endpoint == "batch")).all())
    assert n == 4


def test_batch_limits_and_errors(app_factory, applicant):
    c = app_factory(batch_max_rows=3)
    r = c.post("/v1/score/batch", headers=A, json={"applicants": [applicant] * 4})
    assert r.status_code == 413
    bad = pd.DataFrame([applicant]).drop(columns=["EXT_SOURCE_2"]).assign(EXTRA=1).to_csv(index=False)
    r = c.post("/v1/score/batch", headers={**A, "Content-Type": "text/csv"}, content=bad)
    assert r.status_code == 422
    locs = {e["loc"][-1] for e in r.json()["error"]["errors"]}
    assert locs == {"EXT_SOURCE_2", "EXTRA"}
    r = c.post("/v1/score/batch", headers={**A, "Content-Type": "text/plain"}, content="x")
    assert r.status_code == 415


# ---------------------------------------------------------------- models and hot reload
def test_models_endpoints(client: TestClient):
    lst = client.get("/v1/models", headers=A).json()
    assert lst["serving_version"] == "1" and {v["version"] for v in lst["versions"]} == {"1", "2"}
    assert_provenance(lst)
    m = client.get("/v1/models/1", headers=A).json()
    assert m["result"]["threshold_source"] == "validation" and m["result"]["serving"]
    assert_provenance(m)
    sch = client.get("/v1/models/1/schema", headers=A).json()
    assert [f["importance_rank"] for f in sch["fields"]] == list(range(1, len(sch["fields"]) + 1))
    assert sch["fields"][0]["advanced"] is False and sch["exposure_field"] == "AMT_CREDIT"
    ev = client.get("/v1/models/1/evaluation", headers=A).json()["result"]
    sel = ev["models"][ev["selected"]]
    assert sel["threshold_sweep_validation"] and set(sel["splits"]) == {"validation", "test"}
    card = client.get("/v1/models/2/card", headers=A)
    assert card.status_code == 200 and "Model card" in card.json()["result"]["markdown"]
    assert client.get("/v1/models/99", headers=A).status_code == 404


def test_promote_hot_reloads_without_dropping_requests(client: TestClient, applicant):
    errors: list[int] = []
    stop = threading.Event()

    def hammer():
        while not stop.is_set():
            r = client.post("/v1/score", headers=A, json={"applicant": applicant})
            if r.status_code != 200:
                errors.append(r.status_code)

    threads = [threading.Thread(target=hammer) for _ in range(3)]
    for t in threads:
        t.start()
    r = client.post("/v1/models/2/promote", headers=ADM, json={"alias": "production"})
    stop.set()
    for t in threads:
        t.join()
    assert r.status_code == 200 and r.json()["serving_version"] == "2" and r.json()["replaced"] == "1"
    assert errors == []
    assert client.post("/v1/score", headers=A, json={"applicant": applicant}).json()["model_version"] == "2"
    versions = {v["version"]: v["aliases"] for v in client.get("/v1/models", headers=A).json()["versions"]}
    assert "previous" in versions["1"] and "production" in versions["2"]
    with client.app.state.db.session() as s:
        assert s.scalars(select(AdminEvent).where(AdminEvent.action == "promote_model")).first() is not None
    assert 'creditiq_model_info{model="creditiq-pd",version="2"} 1.0' in client.get("/metrics").text


# ---------------------------------------------------------------- policies
STRICT = (ROOT / "policies" / "2026.10.2-strict.yaml").read_text()


def test_policies_list_validate_simulate(client: TestClient):
    assert {p["version"] for p in client.get("/v1/policies", headers=A).json()} >= {"2026.10.1", "2026.10.2-strict"}
    act = client.get("/v1/policies/active", headers=A).json()
    assert act["summary"]["is_active"] and "annuity_to_income" in act["yaml"]
    bad = client.post("/v1/policies/validate", headers=A, json={"yaml": "version: x\nname: ab\nband_cutoffs: [0.5, 0.1]"}).json()
    assert not bad["valid"] and any("band_cutoffs" in e or "name" in e for e in bad["errors"])
    sens = STRICT.replace("column: AMT_REQ_CREDIT_BUREAU_YEAR", "column: DAYS_BIRTH")
    res = client.post("/v1/policies/validate", headers=A, json={"yaml": sens}).json()
    assert not res["valid"] and any("sensitive" in e or "not an input" in e for e in res["errors"])
    sim = client.post("/v1/policies/simulate", headers=A, json={"yaml": STRICT}).json()
    assert sim["split"] == "validation" and sim["draft"]["flag_rate"] > sim["active"]["flag_rate"]
    assert sim["delta"]["flag_rate"] == pytest.approx(sim["draft"]["flag_rate"] - sim["active"]["flag_rate"])
    assert sum(sim["draft"]["confusion"].values()) == sim["draft"]["n"]
    assert_provenance(sim)
    test = client.post("/v1/policies/simulate", headers=A, json={"yaml": STRICT, "split": "test"}).json()
    assert any("TEST split" in w for w in test["warnings"])
    cmp_ = client.post("/v1/policies/compare", headers=A, json={}).json()
    assert len(cmp_["result"]["policies"]) == 4 and cmp_["result"]["cost_threshold_chosen_on"] == "validation"
    assert "approve" not in json.dumps(cmp_["result"]).lower()
    assert_provenance(cmp_)


def test_admin_saves_and_activates_a_policy(client: TestClient):
    draft = STRICT.replace('version: "2026.10.2-strict"', 'version: "2026.11.0"')
    assert client.post("/v1/policies", headers=A, json={"yaml": draft}).status_code == 403
    r = client.post("/v1/policies", headers=ADM, json={"yaml": draft})
    assert r.status_code == 201 and r.json()["source"] == "ui"
    assert client.post("/v1/policies", headers=ADM, json={"yaml": draft}).status_code == 409  # immutable
    assert client.post("/v1/policies/2026.11.0/activate", headers=ADM).json()["result"]["replaced"] == "2026.10.1"
    assert client.get("/v1/policies/active", headers=A).json()["summary"]["version"] == "2026.11.0"
    assert client.get("/v1/meta", headers=A).json()["policy_version"] == "2026.11.0"


# ---------------------------------------------------------------- analytics
@pytest.mark.parametrize("path", ["/v1/portfolio/summary", "/v1/portfolio/bands", "/v1/portfolio/deciles",
                                  "/v1/portfolio/segments", "/v1/portfolio/concentration",
                                  "/v1/portfolio/summary?lgd=0.6&ccf=0.9&split=validation"])
def test_portfolio(client: TestClient, path):
    r = client.get(path, headers=A)
    assert r.status_code == 200, r.text
    assert_provenance(r.json())


def test_portfolio_numbers_are_consistent(client: TestClient):
    s = client.get("/v1/portfolio/summary", headers=A).json()["summary"]
    bands = client.get("/v1/portfolio/bands", headers=A).json()["rows"]
    assert sum(b["expected_loss"] for b in bands) == pytest.approx(s["total_el"])
    s2 = client.get("/v1/portfolio/summary?lgd=0.9", headers=A).json()["summary"]
    assert s2["total_el"] == pytest.approx(s["total_el"] * 0.9 / 0.45)  # editable LGD drives EL
    assert client.get("/v1/portfolio/segments?column=CODE_GENDER", headers=A).status_code == 422


def test_scenarios(client: TestClient):
    e = client.post("/v1/scenarios/economics", headers=A, json={"pd": 0.05}).json()
    c = e["result"]["contribution"]
    assert c["break_even_pd"] is not None and len(e["result"]["heatmap"]) == 132
    assert_provenance(e)
    st = client.post("/v1/scenarios/stress", headers=A, json={}).json()
    names = [r["scenario"] for r in st["result"]["scenarios"]]
    assert names == ["Improved", "Base", "Adverse", "Severe stress"]
    base = st["result"]["scenarios"][1]["expected_loss"]
    summary = client.get("/v1/portfolio/summary", headers=A).json()["summary"]
    assert base == pytest.approx(summary["total_el"])  # Base scenario = portfolio EL at h = 1
    assert_provenance(st)
    n = client.post("/v1/scenarios/npv", headers=A, json={"lgd": 0.0, "recovery_lag_months": 0,
                                                          "discount_rate": 0.16}).json()["result"]
    assert n["npv_risk_adjusted"] == pytest.approx(n["npv_contractual"], abs=1e-6)  # v1 sanity identity
    assert client.post("/v1/scenarios/npv", headers=A, json={"lgd": 2}).status_code == 422


def test_fairness(client: TestClient):
    r = client.get("/v1/fairness/report", headers=A).json()
    res = r["result"]
    assert set(res["available_attributes"]) >= {"CODE_GENDER", "DAYS_BIRTH"}
    assert all("within_threshold_rate" in g for g in res["groups"]) and res["proxy_scan"]
    assert_provenance(r)
    assert client.get("/v1/fairness/report?attribute=AMT_CREDIT", headers=A).status_code == 422


# ---------------------------------------------------------------- monitoring
def test_monitoring_check_raises_alerts_on_the_shifted_batch(client: TestClient):
    data = (ROOT / "data" / "sample_monitoring_batch.csv").read_bytes()
    r = client.post("/v1/monitoring/check", headers=A, files={"file": ("sample_monitoring_batch.csv", data,
                                                                        "text/csv")})
    assert r.status_code == 200, r.text
    run = r.json()["result"]
    assert run["status"] in ("warning", "critical") and run["alerts"] and run["labels_available"]
    assert {"missingness_jump", "score_shift"} & {a["code"] for a in run["alerts"]}
    assert_provenance(r.json())
    latest = client.get("/v1/monitoring/drift", headers=A).json()["result"]
    assert latest["id"] == run["id"]
    assert client.get("/v1/monitoring/history", headers=A).json()["result"][0]["id"] == run["id"]
    assert client.get("/v1/monitoring/alerts", headers=A).json()["result"]
    metrics = client.get("/metrics").text
    assert "creditiq_drift_feature_psi" in metrics and "creditiq_drift_status" in metrics


# ---------------------------------------------------------------- decisions log, overview, metrics
def test_decisions_log_and_overview(client: TestClient, applicant):
    rid = client.post("/v1/decide", headers=A, json={"applicant": applicant}).json()["request_id"]
    lst = client.get("/v1/decisions?endpoint=decide", headers=A).json()["result"]
    assert lst["total"] >= 1 and lst["items"][0]["request_id"] == rid
    assert client.get(f"/v1/decisions?q={rid[:8]}", headers=A).json()["result"]["total"] == 1
    one = client.get(f"/v1/decisions/{rid}", headers=A).json()
    assert one["result"]["policy_version"] == "2026.10.1" and one["result"]["model_version"] == "1"
    assert_provenance(one)
    assert client.get("/v1/decisions/nope", headers=A).status_code == 404
    ov = client.get("/v1/overview", headers=A).json()
    assert ov["result"]["portfolio_ead"] > 0 and ov["result"]["decisions_total"] >= 1 and ov["result"]["trend"]
    assert_provenance(ov)
    m = client.get("/metrics").text
    for name in ("creditiq_http_request_duration_seconds_bucket", "creditiq_pd_bucket", "creditiq_decisions_total",
                 "creditiq_model_version", "creditiq_policy_info", "creditiq_audit_write_seconds"):
        assert name in m, name


# ---------------------------------------------------------------- migrations
def test_migrations_round_trip_and_match_the_models(tmp_path: Path):
    from alembic.autogenerate import compare_metadata
    from alembic.migration import MigrationContext

    from creditiq_api import migrate
    from creditiq_api.db import Base, make_engine

    url = f"sqlite:///{(tmp_path / 'm.db').as_posix()}"
    migrate.upgrade(url)
    engine = make_engine(url)
    with engine.connect() as conn:
        diff = compare_metadata(MigrationContext.configure(conn), Base.metadata)
    assert diff == [], diff
    migrate.downgrade(url)
    migrate.upgrade(url)
    engine.dispose()


def test_hosted_postgres_urls_get_the_psycopg_driver() -> None:
    from creditiq_api.settings import Settings

    s = Settings(_env_file=None, database_url="postgres://u:p@host/db?sslmode=require")  # type: ignore[call-arg]
    assert s.database_url == "postgresql+psycopg://u:p@host/db?sslmode=require"
    assert Settings(_env_file=None, database_url="sqlite:///x.db").database_url == "sqlite:///x.db"  # type: ignore[call-arg]
