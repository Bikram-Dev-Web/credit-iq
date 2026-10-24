"""Postgres audit store (SQLAlchemy 2). SQLite works too, for tests and quick local runs.

Privacy: the ``decisions`` table stores scores, outcomes, rule results and an HMAC of the inputs,
never the inputs themselves. Explanations are stored as feature names and signed contributions only.
"""
from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, Integer, String, Text, create_engine, event
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker
from sqlalchemy.types import TypeDecorator

JSONType = JSON().with_variant(JSONB(), "postgresql")


class UTCDateTime(TypeDecorator[datetime]):
    """Timezone-aware UTC timestamps on every backend (SQLite returns naive datetimes otherwise)."""

    impl = DateTime(timezone=True)
    cache_ok = True

    def process_bind_param(self, value: datetime | None, dialect: Any) -> datetime | None:
        if value is not None and value.tzinfo is None:
            value = value.replace(tzinfo=UTC)
        return value

    def process_result_value(self, value: datetime | None, dialect: Any) -> datetime | None:
        if value is not None and value.tzinfo is None:
            value = value.replace(tzinfo=UTC)
        return value


def utcnow() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    pass


class Decision(Base):
    __tablename__ = "decisions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    request_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utcnow, index=True)
    endpoint: Mapped[str] = mapped_column(String(32), index=True)
    batch_id: Mapped[str | None] = mapped_column(String(64), index=True, nullable=True)
    api_key_id: Mapped[str] = mapped_column(String(16))
    role: Mapped[str] = mapped_column(String(16))
    model_name: Mapped[str] = mapped_column(String(64))
    model_version: Mapped[str] = mapped_column(String(32), index=True)
    policy_version: Mapped[str] = mapped_column(String(48), index=True)
    pd: Mapped[float] = mapped_column(Float)
    raw_score: Mapped[float] = mapped_column(Float)
    band: Mapped[str] = mapped_column(String(32), index=True)
    outcome: Mapped[str] = mapped_column(String(32), index=True)
    threshold: Mapped[float] = mapped_column(Float)
    threshold_source: Mapped[str] = mapped_column(String(120))
    rules_fired: Mapped[list[dict[str, Any]]] = mapped_column(JSONType, default=list)
    top_reasons: Mapped[list[dict[str, Any]]] = mapped_column(JSONType, default=list)
    ead: Mapped[float] = mapped_column(Float)
    expected_loss: Mapped[float] = mapped_column(Float)
    input_hash: Mapped[str] = mapped_column(String(64), index=True)
    hash_key_id: Mapped[str] = mapped_column(String(16))
    latency_ms: Mapped[float | None] = mapped_column(Float, nullable=True)


class PolicyRecord(Base):
    __tablename__ = "policies"

    version: Mapped[str] = mapped_column(String(48), primary_key=True)
    name: Mapped[str] = mapped_column(String(120))
    content: Mapped[dict[str, Any]] = mapped_column(JSONType)
    content_hash: Mapped[str] = mapped_column(String(32))
    source: Mapped[str] = mapped_column(String(16))  # "file" | "ui"
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utcnow)
    created_by: Mapped[str] = mapped_column(String(32))
    is_active: Mapped[bool] = mapped_column(Boolean, default=False, index=True)


class MonitoringRun(Base):
    __tablename__ = "monitoring_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utcnow, index=True)
    source: Mapped[str] = mapped_column(String(32))  # "scheduled" | "api" | "replay"
    model_name: Mapped[str] = mapped_column(String(64))
    model_version: Mapped[str] = mapped_column(String(32), index=True)
    batch_name: Mapped[str] = mapped_column(String(200))
    n_rows: Mapped[int] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(String(16))  # "ok" | "warning" | "critical"
    max_feature_psi: Mapped[float | None] = mapped_column(Float, nullable=True)
    n_significant: Mapped[int] = mapped_column(Integer, default=0)
    score_psi: Mapped[float | None] = mapped_column(Float, nullable=True)
    mean_pd: Mapped[float | None] = mapped_column(Float, nullable=True)
    ref_mean_pd: Mapped[float | None] = mapped_column(Float, nullable=True)
    labels_available: Mapped[bool] = mapped_column(Boolean, default=False)
    observed_default_rate: Mapped[float | None] = mapped_column(Float, nullable=True)
    roc_auc: Mapped[float | None] = mapped_column(Float, nullable=True)
    features: Mapped[list[dict[str, Any]]] = mapped_column(JSONType, default=list)
    score: Mapped[dict[str, Any] | None] = mapped_column(JSONType, nullable=True)
    schema_check: Mapped[dict[str, Any]] = mapped_column(JSONType, default=dict)
    thresholds: Mapped[dict[str, Any]] = mapped_column(JSONType, default=dict)


class Alert(Base):
    __tablename__ = "alerts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utcnow, index=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("monitoring_runs.id", ondelete="CASCADE"), index=True)
    code: Mapped[str] = mapped_column(String(32))
    severity: Mapped[str] = mapped_column(String(16), index=True)
    message: Mapped[str] = mapped_column(Text)
    value: Mapped[float | None] = mapped_column(Float, nullable=True)
    threshold: Mapped[float | None] = mapped_column(Float, nullable=True)
    columns: Mapped[list[str]] = mapped_column(JSONType, default=list)


class AdminEvent(Base):
    """Who promoted which model or activated which policy, and when."""

    __tablename__ = "admin_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utcnow, index=True)
    actor: Mapped[str] = mapped_column(String(16))
    action: Mapped[str] = mapped_column(String(32))
    detail: Mapped[dict[str, Any]] = mapped_column(JSONType, default=dict)


def make_engine(url: str) -> Engine:
    kwargs: dict[str, Any] = {"pool_pre_ping": True}
    if url.startswith("sqlite"):
        kwargs["connect_args"] = {"check_same_thread": False}
    engine = create_engine(url, **kwargs)
    if url.startswith("sqlite"):
        @event.listens_for(engine, "connect")
        def _fk(dbapi_conn: Any, _: Any) -> None:  # enforce ON DELETE CASCADE in SQLite
            cur = dbapi_conn.cursor()
            cur.execute("PRAGMA foreign_keys=ON")
            cur.execute("PRAGMA journal_mode=WAL")
            cur.close()
    return engine


class Database:
    def __init__(self, url: str):
        self.url = url
        self.engine = make_engine(url)
        self._sessions = sessionmaker(self.engine, expire_on_commit=False)

    @contextmanager
    def session(self) -> Iterator[Session]:
        with self._sessions() as s, s.begin():
            yield s

    def ping(self) -> bool:
        try:
            with self.engine.connect() as conn:
                conn.exec_driver_sql("SELECT 1")
            return True
        except Exception:
            return False
