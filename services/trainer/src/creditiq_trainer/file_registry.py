"""A directory-based model registry: the MLflow registry exported to plain files.

Used where no MLflow server runs (the hosted demo, CI, offline development). Layout::

    <root>/<model_name>/registry.json            {"aliases": {"production": "3", ...}, "versions": {...}}
    <root>/<model_name>/versions/<version>/      a model-version package (creditiq_core.models.package)

The API reads the same layout (``creditiq_api.registry.FileRegistry``).
"""
from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Any

REGISTRY_FILE = "registry.json"


def _load(model_dir: Path) -> dict[str, Any]:
    path = model_dir / REGISTRY_FILE
    if path.exists():
        data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        return data
    return {"aliases": {}, "versions": {}}


def _save(model_dir: Path, data: dict[str, Any]) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    tmp = model_dir / (REGISTRY_FILE + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(model_dir / REGISTRY_FILE)  # atomic on the same filesystem


def add_version(root: Path, model_name: str, package_dir: Path, version: str | None = None,
                aliases: list[str] | None = None, info: dict[str, Any] | None = None) -> str:
    """Copy a package into the registry as a new (or given) version and set aliases."""
    model_dir = root / model_name
    data = _load(model_dir)
    if version is None:
        version = str(max((int(v) for v in data["versions"]), default=0) + 1)
    dst = model_dir / "versions" / version
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(package_dir, dst)
    data["versions"][version] = {"created": int(time.time() * 1000), **(info or {})}
    for alias in aliases or []:
        set_alias(data, alias, version)
    _save(model_dir, data)
    return version


def set_alias(data: dict[str, Any], alias: str, version: str) -> None:
    old = data["aliases"].get(alias)
    if alias == "production" and old and old != version:
        data["aliases"]["previous"] = old
    data["aliases"][alias] = version
