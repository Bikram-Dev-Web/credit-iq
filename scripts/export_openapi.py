"""Write the API's OpenAPI document (with the served model's Applicant schema) to docs/api/openapi.json.

The web app's typed client is generated from this file (``pnpm gen:api``) and CI fails if it is stale.

Usage:  uv run python scripts/export_openapi.py [--registry models/registry] [--check]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "api" / "openapi.json"


def build(registry: Path) -> dict:
    from fastapi.testclient import TestClient

    from creditiq_api.main import create_app
    from creditiq_api.settings import Settings

    with tempfile.TemporaryDirectory() as tmp:
        settings = Settings(_env_file=None, environment="test", database_url=f"sqlite:///{Path(tmp, 'o.db').as_posix()}",  # type: ignore[call-arg]
                            file_registry_path=str(registry), policies_dir=str(ROOT / "policies"),
                            registry_poll_seconds=0, log_level="WARNING", api_keys="analyst:export-only-key")
        app = create_app(settings)
        with TestClient(app) as client:
            spec: dict = client.get("/openapi.json").json()
        app.state.db.engine.dispose()
    return spec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--registry", type=Path, default=ROOT / "models" / "registry")
    ap.add_argument("--check", action="store_true", help="Exit 1 if docs/api/openapi.json is out of date")
    args = ap.parse_args()
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    text = json.dumps(build(args.registry), indent=2, sort_keys=True) + "\n"
    if args.check:
        if not OUT.exists() or OUT.read_text(encoding="utf-8") != text:
            print("docs/api/openapi.json is stale: run `uv run python scripts/export_openapi.py`", file=sys.stderr)
            return 1
        return 0
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(text, encoding="utf-8", newline="\n")
    print(f"wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
