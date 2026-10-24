"""The served model: load a registry version, build its request schema, swap it in atomically.

``ModelHolder.current`` is an immutable ``LoadedModel``. A promotion (or the registry poller seeing
the alias move) loads the new version in the background, warms it with one prediction, then
replaces the reference in a single assignment. Requests already running keep the object they
started with, so no request is dropped or served by a half-loaded model.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from functools import cached_property
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, create_model

from creditiq_api.registry import Registry
from creditiq_core._frames import iter_rows
from creditiq_core.finance.portfolio import assign_risk_band
from creditiq_core.models import package as pkg
from creditiq_core.models.artifact import EvaluationSnapshot, ModelArtifact

log = logging.getLogger(__name__)


def build_applicant_model(schema: dict[str, dict[str, Any]], name: str = "Applicant") -> type[BaseModel]:
    """Pydantic model generated from the artifact's input schema.

    Every model input must be present (a missing field is a 422); ``null`` explicitly means "unknown" and is
    imputed from the training data, as in v1. Unknown fields are rejected. Categorical values outside the
    training categories are accepted (they map to the encoder's infrequent bucket) and reported as a warning.
    """
    fields: dict[str, Any] = {}
    for col, info in schema.items():
        extra: dict[str, Any] = {"x-missing-rate": round(float(info.get("missing_rate", 0.0)), 4)}
        if info["kind"] == "numeric":
            extra.update({k: info[k] for k in ("min", "max", "p01", "p99", "median", "is_integer") if k in info})
            fields[col] = (float | None, Field(..., description=f"Numeric input. Training median "
                                                                  f"{info.get('median', 0):,.4g}.",
                                                 json_schema_extra=extra))
        else:
            cats = list(info.get("categories", []))
            extra.update({"x-categories": cats, "x-mode": info.get("mode")})
            fields[col] = (str | None, Field(..., max_length=200, description="Categorical input.",
                                             json_schema_extra=extra))
    model: type[BaseModel] = create_model(name, __config__=ConfigDict(extra="forbid"), **fields)
    return model


@dataclass(frozen=True)
class LoadedModel:
    name: str
    version: str
    artifact: ModelArtifact
    snapshot: EvaluationSnapshot
    evaluation: dict[str, Any]
    metadata: dict[str, Any]
    model_card: str
    request_model: type[BaseModel]
    loaded_at: datetime
    package_dir: Path
    _band_cache: dict[tuple[float, ...], dict[str, dict[str, float]]] = field(default_factory=dict, compare=False)

    @property
    def is_synthetic(self) -> bool:
        return self.artifact.is_synthetic

    @property
    def key(self) -> str:
        return str(self.metadata.get("artifact_model") or self.metadata.get("selected_model"))

    @property
    def label(self) -> str:
        return str(self.evaluation["models"][self.key]["label"])

    @cached_property
    def test_metrics(self) -> dict[str, Any]:
        m: dict[str, Any] = self.evaluation["models"][self.key]["splits"]["test"]["metrics"]
        return m

    @cached_property
    def test_ece(self) -> float | None:
        v = self.evaluation["models"][self.key]["splits"]["test"].get("ece")
        return None if v is None else float(v)

    @cached_property
    def importance(self) -> dict[str, float]:
        """Model-internal importance per input (derived ratios also credit their source inputs), for field order."""
        from creditiq_core.explainability import native_importance
        from creditiq_core.features.engineering import DOMAIN_FEATURES

        table, _ = native_importance(self.artifact.model, self.artifact.transformed_mean)
        imp = {str(r.feature): float(r.importance) for r in iter_rows(table)}
        for feat, (sources, _desc) in DOMAIN_FEATURES.items():
            if feat in imp:
                for src in sources:
                    if src in self.input_columns:
                        imp[src] = imp.get(src, 0.0) + imp[feat] / len(sources)
        return imp

    @cached_property
    def input_columns(self) -> list[str]:
        return self.artifact.input_columns

    @cached_property
    def exposure_col(self) -> str | None:
        return self.snapshot.exposure_col

    def band_reliability(self, cutoffs: tuple[float, ...], labels: tuple[str, ...]) -> dict[str, dict[str, float]]:
        """Observed default rate and count per risk band on the held-out TEST split (for the uncertainty panel)."""
        key = (*cutoffs, len(labels))
        if key not in self._band_cache:
            test = self.snapshot.split("test")
            bands = assign_risk_band(test["pd"].to_numpy(), cutoffs, labels).astype(str)
            frame = pd.DataFrame({"band": bands.to_numpy(), "y": test["y"].to_numpy(), "pd": test["pd"].to_numpy()})
            g = frame.groupby("band")
            self._band_cache[key] = {str(b): {"n": int(len(d)), "observed_default_rate": float(d["y"].mean()),
                                              "mean_pd": float(d["pd"].mean())} for b, d in g}
        return self._band_cache[key]


def load_package(name: str, version: str, path: Path) -> LoadedModel:
    artifact = pkg.artifact_from_bytes((path / pkg.ARTIFACT_FILE).read_bytes())
    frame = pd.read_parquet(path / pkg.SNAPSHOT_FILE)
    snapshot = pkg.snapshot_from_parts(frame, json.loads((path / pkg.SNAPSHOT_META_FILE).read_text("utf-8")))
    evaluation = json.loads((path / pkg.EVALUATION_FILE).read_text("utf-8"))
    metadata = json.loads((path / pkg.METADATA_FILE).read_text("utf-8"))
    card = (path / pkg.MODEL_CARD_FILE).read_text("utf-8")
    model = LoadedModel(name=name, version=str(version), artifact=artifact, snapshot=snapshot, evaluation=evaluation,
                        metadata=metadata, model_card=card,
                        request_model=build_applicant_model(artifact.input_schema),
                        loaded_at=datetime.now(UTC), package_dir=path)
    # Warm-up: one prediction (and the lazy band table) before the model can be served.
    probe = {c: None for c in artifact.input_columns}
    p = float(artifact.predict_pd(probe)[0])
    if not np.isfinite(p):
        raise ValueError(f"{name} v{version} returned a non-finite PD on the warm-up row")
    return model


class ModelHolder:
    def __init__(self, registry: Registry, alias: str, poll_seconds: float = 30.0):
        self.registry = registry
        self.alias = alias
        self.poll_seconds = poll_seconds
        self._current: LoadedModel | None = None
        self._load_lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.last_error: str | None = None
        self.reloads = 0
        self.listeners: list[Any] = []

    @property
    def current(self) -> LoadedModel | None:
        return self._current

    def require(self) -> LoadedModel:
        m = self._current
        if m is None:
            raise ModelNotLoaded(self.last_error or f"No model version has the '{self.alias}' alias yet.")
        return m

    def load_version(self, version: str, reason: Literal["startup", "promotion", "poll", "manual"]) -> LoadedModel:
        with self._load_lock:
            cur = self._current
            if cur is not None and cur.version == str(version):
                return cur
            t0 = time.perf_counter()
            model = load_package(self.registry.model_name, str(version), self.registry.fetch(str(version)))
            self._current = model  # atomic reference swap
            self.reloads += 1
            self.last_error = None
            log.info("model loaded", extra={"model_version": model.version, "reason": reason,
                                            "load_seconds": round(time.perf_counter() - t0, 3)})
        for fn in self.listeners:
            fn(model)
        return model

    def refresh(self, reason: Literal["startup", "promotion", "poll", "manual"] = "poll") -> LoadedModel | None:
        """Load whatever the alias points to now (no-op if unchanged)."""
        try:
            version = self.registry.resolve(self.alias)
            if version is None:
                self.last_error = f"No version of {self.registry.model_name} has the '{self.alias}' alias."
                return self._current
            return self.load_version(version, reason)
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            log.exception("model refresh failed")
            return self._current

    def start_polling(self) -> None:
        if self.poll_seconds <= 0 or self._thread is not None:
            return

        def loop() -> None:
            while not self._stop.wait(self.poll_seconds):
                self.refresh("poll")

        self._thread = threading.Thread(target=loop, name="registry-poller", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()


class ModelNotLoaded(RuntimeError):
    pass
