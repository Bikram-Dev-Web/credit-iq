"""Model registries the API can serve from: MLflow, or a file registry exported from it.

Both expose the same small interface: list versions, resolve an alias, fetch a version's package
directory and move an alias. The file layout matches ``creditiq_trainer.file_registry``.
"""
from __future__ import annotations

import json
import os
import shutil
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

REGISTRY_FILE = "registry.json"


@dataclass
class VersionInfo:
    version: str
    aliases: list[str]
    created: int | None = None
    description: str | None = None
    run_id: str | None = None
    info: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"version": self.version, "aliases": self.aliases, "created": self.created,
                "description": self.description, "run_id": self.run_id, **self.info}


class Registry(Protocol):
    model_name: str

    def versions(self) -> list[VersionInfo]: ...
    def resolve(self, alias: str) -> str | None: ...
    def fetch(self, version: str) -> Path: ...
    def set_alias(self, alias: str, version: str) -> dict[str, str | None]: ...


class FileRegistry:
    def __init__(self, root: str | Path, model_name: str):
        self.root = Path(root)
        self.model_name = model_name
        self._lock = threading.Lock()

    @property
    def _dir(self) -> Path:
        return self.root / self.model_name

    def _load(self) -> dict[str, Any]:
        path = self._dir / REGISTRY_FILE
        if not path.exists():
            return {"aliases": {}, "versions": {}}
        data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        return data

    def versions(self) -> list[VersionInfo]:
        data = self._load()
        by_version: dict[str, list[str]] = {}
        for alias, v in data["aliases"].items():
            by_version.setdefault(str(v), []).append(alias)
        out = []
        for v, info in sorted(data["versions"].items(), key=lambda kv: int(kv[0]), reverse=True):
            info = dict(info)
            out.append(VersionInfo(version=v, aliases=sorted(by_version.get(v, [])), created=info.pop("created", None),
                                   description=info.pop("description", None), run_id=info.pop("run_id", None),
                                   info=info))
        return out

    def resolve(self, alias: str) -> str | None:
        v = self._load()["aliases"].get(alias)
        return str(v) if v is not None else None

    def fetch(self, version: str) -> Path:
        path = self._dir / "versions" / str(version)
        if not path.exists():
            raise KeyError(f"{self.model_name} has no version {version}")
        return path

    def set_alias(self, alias: str, version: str) -> dict[str, str | None]:
        with self._lock:
            data = self._load()
            if str(version) not in data["versions"]:
                raise KeyError(f"{self.model_name} has no version {version}")
            old = data["aliases"].get(alias)
            if alias == "production" and old and old != version:
                data["aliases"]["previous"] = old
            data["aliases"][alias] = str(version)
            tmp = self._dir / (REGISTRY_FILE + ".tmp")
            tmp.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
            tmp.replace(self._dir / REGISTRY_FILE)
        return {"alias": alias, "version": str(version), "replaced": old}


class MlflowRegistry:
    def __init__(self, tracking_uri: str | None, model_name: str, cache_dir: str | Path):
        os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
        import mlflow
        from mlflow.tracking import MlflowClient

        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        self._mlflow = mlflow
        self.client = MlflowClient()
        self.model_name = model_name
        self.cache = Path(cache_dir)

    def versions(self) -> list[VersionInfo]:
        try:
            reg = self.client.get_registered_model(self.model_name)
            mvs = self.client.search_model_versions(f"name = '{self.model_name}'")
        except self._mlflow.exceptions.MlflowException:
            return []
        by_version: dict[str, list[str]] = {}
        for alias, v in (reg.aliases or {}).items():
            by_version.setdefault(str(v), []).append(alias)
        out = []
        for mv in sorted(mvs, key=lambda m: int(m.version), reverse=True):
            tags = dict(mv.tags or {})
            out.append(VersionInfo(version=str(mv.version), aliases=sorted(by_version.get(str(mv.version), [])),
                                   created=mv.creation_timestamp, description=mv.description, run_id=mv.run_id,
                                   info={"is_synthetic": tags.get("creditiq.is_synthetic") == "true",
                                         "dataset": tags.get("creditiq.dataset_name"),
                                         "selected_model": tags.get("creditiq.selected_model")}))
        return out

    def resolve(self, alias: str) -> str | None:
        try:
            return str(self.client.get_model_version_by_alias(self.model_name, alias).version)
        except self._mlflow.exceptions.MlflowException:
            return None

    def fetch(self, version: str) -> Path:
        dst = self.cache / self.model_name / str(version)
        marker = dst / ".complete"
        if not marker.exists():
            if dst.exists():
                shutil.rmtree(dst)
            dst.mkdir(parents=True)
            local = Path(self._mlflow.artifacts.download_artifacts(
                artifact_uri=f"models:/{self.model_name}/{version}", dst_path=str(dst)))
            if next(local.rglob("model_artifact.joblib"), None) is None:
                raise FileNotFoundError(f"{self.model_name} v{version} does not contain a CreditIQ package")
            marker.touch()
        found = next(dst.rglob("model_artifact.joblib"))
        return found.parent

    def set_alias(self, alias: str, version: str) -> dict[str, str | None]:
        self.client.get_model_version(self.model_name, version)
        old = self.resolve(alias)
        if alias == "production" and old and old != version:
            self.client.set_registered_model_alias(self.model_name, "previous", old)
        self.client.set_registered_model_alias(self.model_name, alias, version)
        return {"alias": alias, "version": str(version), "replaced": old}
