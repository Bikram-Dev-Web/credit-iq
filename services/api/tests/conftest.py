"""API test fixtures: two real (small) model versions in a file registry, SQLite audit store, repo policies."""
from __future__ import annotations

import os
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from fastapi.testclient import TestClient

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

from creditiq_api.main import create_app
from creditiq_api.settings import Settings
from creditiq_trainer import file_registry, pipeline
from creditiq_trainer.config import load_config
from creditiq_trainer.package_writer import write_package

ROOT = Path(__file__).resolve().parents[3]
ANALYST, ADMIN = "test-analyst-key-123", "test-admin-key-456"
A = {"X-API-Key": ANALYST}
ADM = {"X-API-Key": ADMIN}


@pytest.fixture(scope="session")
def registry_root() -> Path:
    """v1 (production): logreg + gbm; v2 (staging): logreg only with another seed."""
    root = Path(tempfile.mkdtemp(prefix="creditiq_reg_"))
    base = load_config(ROOT / "services/trainer/configs/synthetic_ci.yaml")
    base.dataset.path = str(ROOT / "data/sample_credit_data.csv")
    for version, alias, update in (("1", "production", {"sample_size": 4000}),
                                   ("2", "staging", {"sample_size": 3000, "seed": 11, "models": ["logreg"]})):
        cfg = base.model_copy(update=update)
        data = pipeline.prepare(cfg)
        result = pipeline.train(cfg, data)
        pkg = write_package(cfg, data, result, root / f"pkg{version}")
        file_registry.add_version(root, "creditiq-pd", pkg, version=version, aliases=[alias],
                                  info={"selected_model": result.selected, "is_synthetic": True,
                                        "test_roc_auc": result.metrics[result.selected]["test"]["roc_auc"]})
    return root


def make_settings(tmp: Path, registry_root: Path, **over: Any) -> Settings:
    values: dict[str, Any] = {
        "environment": "test", "database_url": f"sqlite:///{(tmp / 'audit.db').as_posix()}",
        "registry_backend": "file", "file_registry_path": str(registry_root), "registry_poll_seconds": 0,
        "policies_dir": str(ROOT / "policies"), "active_policy": "2026.10.1",
        "audit_hash_key": "test-hash-key-0123456789", "api_keys": f"analyst:{ANALYST},admin:{ADMIN}",
        "log_json": True, "rate_limit_score": "100000/minute",
    }
    values.update(over)
    return Settings(_env_file=None, **values)  # type: ignore[call-arg]


@pytest.fixture()
def app_factory(tmp_path: Path, registry_root: Path) -> Iterator[Any]:
    clients: list[TestClient] = []

    def factory(**over: Any) -> TestClient:
        # Each app gets its own copy of the registry so promotions do not leak between tests.
        import shutil

        reg = tmp_path / f"reg{len(clients)}"
        shutil.copytree(registry_root, reg)
        client = TestClient(create_app(make_settings(tmp_path / f"db{len(clients)}", reg, **over)))
        client.__enter__()
        clients.append(client)
        return client

    yield factory
    for c in clients:
        c.__exit__(None, None, None)


@pytest.fixture()
def client(app_factory: Any) -> TestClient:
    c: TestClient = app_factory()
    return c


@pytest.fixture()
def applicant(client: TestClient) -> dict[str, Any]:
    """A real held-out applicant from the served model's evaluation snapshot (inputs only)."""
    model = client.app.state.holder.current  # type: ignore[attr-defined]
    row = model.snapshot.split("test").iloc[0]
    out = {}
    for c in model.input_columns:
        v = row[c]
        out[c] = None if pd.isna(v) else (v.item() if hasattr(v, "item") else v)
    return out


@pytest.fixture(scope="session")
def settings_for() -> Any:
    """``settings_for(tmp_dir, registry_root, **overrides)`` for tests that build their own app."""
    return make_settings
