"""YAML training configuration, validated with Pydantic.

The target mapping has no default on purpose: CreditIQ never assumes that ``1`` means default.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from creditiq_core import defaults


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DatasetConfig(_Strict):
    path: str = Field(description="CSV file, relative to the repository root or absolute")
    kind: Literal["synthetic", "home_credit", "csv"] = "csv"
    load_rows: int | None = Field(default=None, gt=0, description="Randomly sample this many rows while reading")


class TargetConfig(_Strict):
    column: str
    default_values: list[int | float | str] = Field(min_length=1, description="Label value(s) that mean DEFAULT")
    non_default_values: list[int | float | str] | None = Field(
        default=None, description="Label value(s) that mean NON-DEFAULT (default: every other observed label)")
    definition: str | None = None


class ColumnsConfig(_Strict):
    exclude: list[str] = Field(default_factory=list, description="Extra columns to keep out of the model")
    sensitive: list[str] = Field(default_factory=list,
                                 description="Sensitive attributes for the fairness audit (merged with auto-detected)")
    date: str | None = None
    exposure: str | None = None
    keep_sensitive_in_model: bool = False


class SplitConfig(_Strict):
    strategy: Literal["stratified", "chronological"] = "stratified"
    validation: float = Field(default=0.2, gt=0, lt=0.5)
    test: float = Field(default=0.2, gt=0, lt=0.5)


class ThresholdConfig(_Strict):
    rule: Literal["max_f1", "youden", "prevalence", "manual"] = "max_f1"
    manual: float = Field(default=0.5, gt=0, lt=1)


class LeakageGuardConfig(_Strict):
    enabled: bool = True
    auc_threshold: float = Field(default=0.90, gt=0.5, le=1.0)


class RegistryConfig(_Strict):
    model_name: str = "creditiq-pd"
    enabled: bool = Field(default=True, description="Register the selected model as a new version")
    alias: str | None = Field(default="staging", description="Alias given to the new version (None: no alias)")


class TrainingFileConfig(_Strict):
    name: str
    description: str = ""
    dataset: DatasetConfig
    target: TargetConfig
    columns: ColumnsConfig = Field(default_factory=ColumnsConfig)
    split: SplitConfig = Field(default_factory=SplitConfig)
    sample_size: int | None = Field(default=defaults.DEFAULT_TRAIN_SAMPLE, gt=0)
    models: list[Literal["logreg", "random_forest", "gbm"]] = Field(
        default=["logreg", "random_forest", "gbm"], min_length=1)
    gbm_backend: Literal["lightgbm", "xgboost", "sklearn_hgb"] = "lightgbm"
    params: dict[str, dict[str, Any]] = Field(default_factory=dict)
    class_weight: Literal["balanced", "none"] = "balanced"
    calibration: Literal["sigmoid", "isotonic", "none"] = "sigmoid"
    threshold: ThresholdConfig = Field(default_factory=ThresholdConfig)
    selection_metric: Literal["roc_auc", "pr_auc", "brier", "log_loss"] = "roc_auc"
    tune: bool = False
    use_domain_features: bool = True
    leakage_guard: LeakageGuardConfig = Field(default_factory=LeakageGuardConfig)
    seed: int = defaults.DEFAULT_SEED
    registry: RegistryConfig = Field(default_factory=RegistryConfig)

    @field_validator("params")
    @classmethod
    def _params_keys(cls, v: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
        unknown = set(v) - {"logreg", "random_forest", "gbm"}
        if unknown:
            raise ValueError(f"params has unknown model keys: {sorted(unknown)}")
        return v

    @model_validator(mode="after")
    def _checks(self) -> TrainingFileConfig:
        if self.split.validation + self.split.test >= 0.8:
            raise ValueError("split.validation + split.test must leave at least 20% for training")
        if self.split.strategy == "chronological" and not self.columns.date:
            raise ValueError("a chronological split needs columns.date")
        if self.target.column in self.columns.exclude:
            raise ValueError("the target column is excluded automatically; do not list it in columns.exclude")
        return self

    def config_hash(self) -> str:
        """Stable hash of the effective configuration (recorded on every run)."""
        return hashlib.sha256(json.dumps(self.model_dump(mode="json"), sort_keys=True).encode()).hexdigest()[:12]


class ConfigError(ValueError):
    """A training config that cannot be read or does not validate."""


def load_config(path: str | Path) -> TrainingFileConfig:
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"Config file not found: {path}")
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path.name} is not valid YAML: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError(f"{path.name} must contain a YAML mapping at the top level")
    try:
        return TrainingFileConfig.model_validate(raw)
    except ValueError as exc:
        raise ConfigError(f"{path.name} is not a valid training config:\n{exc}") from exc
