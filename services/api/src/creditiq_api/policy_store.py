"""Policies: YAML files in ``policies/`` are the reviewed source; Postgres holds runtime versions.

At start-up every YAML file is validated and seeded into the ``policies`` table (an existing version
with different content is an error: published versions are immutable). Versions saved from the UI
are stored with ``source = "ui"`` and can be exported as YAML for review. Exactly one policy is active.
"""
from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any

import yaml
from sqlalchemy import select, update

from creditiq_api.db import AdminEvent, Database, PolicyRecord
from creditiq_core.policy.schema import Policy, PolicyValidationError, parse_policy

log = logging.getLogger(__name__)


class PolicyConflict(ValueError):
    pass


class PolicyStore:
    def __init__(self, db: Database, policies_dir: str | Path, default_active: str | None = None):
        self.db = db
        self.dir = Path(policies_dir)
        self.default_active = default_active
        self._active: Policy | None = None
        self._lock = threading.Lock()
        self.load_errors: dict[str, list[str]] = {}

    # ------------------------------------------------------------------ start-up
    def seed(self) -> None:
        files = sorted(self.dir.glob("*.yaml")) if self.dir.exists() else []
        with self.db.session() as s:
            for path in files:
                try:
                    policy = parse_policy(yaml.safe_load(path.read_text(encoding="utf-8")))
                except PolicyValidationError as exc:
                    self.load_errors[path.name] = exc.errors
                    log.error("invalid policy file skipped", extra={"file": path.name, "errors": exc.errors})
                    continue
                existing = s.get(PolicyRecord, policy.version)
                if existing is None:
                    s.add(PolicyRecord(version=policy.version, name=policy.name,
                                       content=policy.model_dump(mode="json"), content_hash=policy.content_hash(),
                                       source="file", created_by="seed", is_active=False))
                elif existing.content_hash != policy.content_hash():
                    msg = (f"{path.name}: version {policy.version} already exists with different content; published "
                           "policy versions are immutable — give the edited policy a new version")
                    self.load_errors[path.name] = [msg]
                    log.error(msg)
            if s.scalar(select(PolicyRecord).where(PolicyRecord.is_active.is_(True))) is None:
                wanted = self.default_active
                rec = s.get(PolicyRecord, wanted) if wanted else None
                if rec is None:
                    rec = s.scalars(select(PolicyRecord).order_by(PolicyRecord.version)).first()
                if rec is not None:
                    rec.is_active = True
        self._active = None
        self.active()

    # ------------------------------------------------------------------ reads
    def active(self) -> Policy:
        if self._active is None:
            with self._lock, self.db.session() as s:
                rec = s.scalar(select(PolicyRecord).where(PolicyRecord.is_active.is_(True)))
                if rec is None:
                    raise LookupError("No active lending policy. Add a policy YAML file to policies/.")
                self._active = Policy.model_validate(rec.content)
        return self._active

    def list(self) -> list[dict[str, Any]]:
        with self.db.session() as s:
            recs = s.scalars(select(PolicyRecord).order_by(PolicyRecord.created_at.desc())).all()
            return [self._summary(r) for r in recs]

    def get(self, version: str) -> tuple[Policy, dict[str, Any]]:
        with self.db.session() as s:
            rec = s.get(PolicyRecord, version)
            if rec is None:
                raise KeyError(version)
            return Policy.model_validate(rec.content), self._summary(rec)

    @staticmethod
    def _summary(r: PolicyRecord) -> dict[str, Any]:
        return {"version": r.version, "name": r.name, "source": r.source, "created_at": r.created_at,
                "created_by": r.created_by, "is_active": r.is_active, "content_hash": r.content_hash,
                "parent_version": r.content.get("parent_version"), "description": r.content.get("description", "")}

    # ------------------------------------------------------------------ admin writes
    def create(self, policy: Policy, actor: str) -> dict[str, Any]:
        with self.db.session() as s:
            if s.get(PolicyRecord, policy.version) is not None:
                raise PolicyConflict(f"Policy version {policy.version} already exists; versions are immutable.")
            rec = PolicyRecord(version=policy.version, name=policy.name, content=policy.model_dump(mode="json"),
                               content_hash=policy.content_hash(), source="ui", created_by=actor, is_active=False)
            s.add(rec)
            s.add(AdminEvent(actor=actor, action="save_policy", detail={"version": policy.version}))
            s.flush()
            return self._summary(rec)

    def activate(self, version: str, actor: str) -> dict[str, Any]:
        with self.db.session() as s:
            rec = s.get(PolicyRecord, version)
            if rec is None:
                raise KeyError(version)
            old = s.scalar(select(PolicyRecord.version).where(PolicyRecord.is_active.is_(True)))
            s.execute(update(PolicyRecord).values(is_active=False))
            rec.is_active = True
            s.add(AdminEvent(actor=actor, action="activate_policy", detail={"version": version, "replaced": old}))
            policy = Policy.model_validate(rec.content)
        self._active = policy
        return {"version": version, "replaced": old}

    @staticmethod
    def to_yaml(policy: Policy) -> str:
        return yaml.safe_dump(policy.model_dump(mode="json", exclude_defaults=False), sort_keys=False)
