"""Migrations and the audit log on a real Postgres (CI service container). Skipped unless TEST_POSTGRES_URL is set,
e.g. postgresql+psycopg://creditiq:creditiq@localhost:5432/creditiq_test."""
from __future__ import annotations

import os

import pytest
from sqlalchemy import select, text

URL = os.environ.get("TEST_POSTGRES_URL")
pytestmark = pytest.mark.skipif(not URL, reason="TEST_POSTGRES_URL not set")


def test_postgres_migrations_and_jsonb_round_trip() -> None:
    from alembic.autogenerate import compare_metadata
    from alembic.migration import MigrationContext

    from creditiq_api import migrate
    from creditiq_api.db import Base, Database, Decision

    assert URL
    migrate.downgrade(URL)
    migrate.upgrade(URL)
    db = Database(URL)
    with db.engine.connect() as conn:
        assert compare_metadata(MigrationContext.configure(conn), Base.metadata) == []
        kind = conn.execute(text("SELECT data_type FROM information_schema.columns WHERE table_name='decisions' "
                                 "AND column_name='rules_fired'")).scalar()
    assert kind == "jsonb"
    with db.session() as s:
        s.add(Decision(request_id="pg-1", endpoint="score", api_key_id="k", role="analyst", model_name="m",
                       model_version="1", policy_version="p", pd=0.1, raw_score=0.2, band="Moderate",
                       outcome="within_threshold", threshold=0.2, threshold_source="model",
                       rules_fired=[{"id": "r"}], top_reasons=[{"feature": "x", "contribution": 0.1}], ead=1.0,
                       expected_loss=0.045, input_hash="h" * 64, hash_key_id="k1"))
    with db.session() as s:
        row = s.scalar(select(Decision).where(Decision.request_id == "pg-1"))
        assert row is not None and row.rules_fired == [{"id": "r"}] and row.created_at.tzinfo is not None
    db.engine.dispose()
