"""Initial schema: decisions (audit log), policies, monitoring runs, alerts, admin events.

Revision ID: 0001
Revises:
Create Date: 2026-10-24
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None

JSON = sa.JSON().with_variant(JSONB(), "postgresql")
TS = sa.DateTime(timezone=True)


def upgrade() -> None:
    op.create_table(
        "decisions",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("request_id", sa.String(64), nullable=False),
        sa.Column("created_at", TS, nullable=False),
        sa.Column("endpoint", sa.String(32), nullable=False),
        sa.Column("batch_id", sa.String(64), nullable=True),
        sa.Column("api_key_id", sa.String(16), nullable=False),
        sa.Column("role", sa.String(16), nullable=False),
        sa.Column("model_name", sa.String(64), nullable=False),
        sa.Column("model_version", sa.String(32), nullable=False),
        sa.Column("policy_version", sa.String(48), nullable=False),
        sa.Column("pd", sa.Float(), nullable=False),
        sa.Column("raw_score", sa.Float(), nullable=False),
        sa.Column("band", sa.String(32), nullable=False),
        sa.Column("outcome", sa.String(32), nullable=False),
        sa.Column("threshold", sa.Float(), nullable=False),
        sa.Column("threshold_source", sa.String(120), nullable=False),
        sa.Column("rules_fired", JSON, nullable=False),
        sa.Column("top_reasons", JSON, nullable=False),
        sa.Column("ead", sa.Float(), nullable=False),
        sa.Column("expected_loss", sa.Float(), nullable=False),
        sa.Column("input_hash", sa.String(64), nullable=False),
        sa.Column("hash_key_id", sa.String(16), nullable=False),
        sa.Column("latency_ms", sa.Float(), nullable=True),
    )
    for col, unique in (("request_id", True), ("created_at", False), ("endpoint", False), ("batch_id", False),
                        ("model_version", False), ("policy_version", False), ("band", False), ("outcome", False),
                        ("input_hash", False)):
        op.create_index(f"ix_decisions_{col}", "decisions", [col], unique=unique)

    op.create_table(
        "policies",
        sa.Column("version", sa.String(48), primary_key=True),
        sa.Column("name", sa.String(120), nullable=False),
        sa.Column("content", JSON, nullable=False),
        sa.Column("content_hash", sa.String(32), nullable=False),
        sa.Column("source", sa.String(16), nullable=False),
        sa.Column("created_at", TS, nullable=False),
        sa.Column("created_by", sa.String(32), nullable=False),
        sa.Column("is_active", sa.Boolean(), nullable=False),
    )
    op.create_index("ix_policies_is_active", "policies", ["is_active"])

    op.create_table(
        "monitoring_runs",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("created_at", TS, nullable=False),
        sa.Column("source", sa.String(32), nullable=False),
        sa.Column("model_name", sa.String(64), nullable=False),
        sa.Column("model_version", sa.String(32), nullable=False),
        sa.Column("batch_name", sa.String(200), nullable=False),
        sa.Column("n_rows", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("max_feature_psi", sa.Float(), nullable=True),
        sa.Column("n_significant", sa.Integer(), nullable=False),
        sa.Column("score_psi", sa.Float(), nullable=True),
        sa.Column("mean_pd", sa.Float(), nullable=True),
        sa.Column("ref_mean_pd", sa.Float(), nullable=True),
        sa.Column("labels_available", sa.Boolean(), nullable=False),
        sa.Column("observed_default_rate", sa.Float(), nullable=True),
        sa.Column("roc_auc", sa.Float(), nullable=True),
        sa.Column("features", JSON, nullable=False),
        sa.Column("score", JSON, nullable=True),
        sa.Column("schema_check", JSON, nullable=False),
        sa.Column("thresholds", JSON, nullable=False),
    )
    op.create_index("ix_monitoring_runs_created_at", "monitoring_runs", ["created_at"])
    op.create_index("ix_monitoring_runs_model_version", "monitoring_runs", ["model_version"])

    op.create_table(
        "alerts",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("created_at", TS, nullable=False),
        sa.Column("run_id", sa.Integer(), sa.ForeignKey("monitoring_runs.id", ondelete="CASCADE"), nullable=False),
        sa.Column("code", sa.String(32), nullable=False),
        sa.Column("severity", sa.String(16), nullable=False),
        sa.Column("message", sa.Text(), nullable=False),
        sa.Column("value", sa.Float(), nullable=True),
        sa.Column("threshold", sa.Float(), nullable=True),
        sa.Column("columns", JSON, nullable=False),
    )
    op.create_index("ix_alerts_created_at", "alerts", ["created_at"])
    op.create_index("ix_alerts_run_id", "alerts", ["run_id"])
    op.create_index("ix_alerts_severity", "alerts", ["severity"])

    op.create_table(
        "admin_events",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("created_at", TS, nullable=False),
        sa.Column("actor", sa.String(16), nullable=False),
        sa.Column("action", sa.String(32), nullable=False),
        sa.Column("detail", JSON, nullable=False),
    )
    op.create_index("ix_admin_events_created_at", "admin_events", ["created_at"])


def downgrade() -> None:
    for table in ("admin_events", "alerts", "monitoring_runs", "policies", "decisions"):
        op.drop_table(table)
