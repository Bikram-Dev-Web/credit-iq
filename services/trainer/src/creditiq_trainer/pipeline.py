"""Training pipeline: read data → confirm target mapping → feature roles → train (creditiq_core)."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from creditiq_core.data import csv as csvio
from creditiq_core.data import validation as dv
from creditiq_core.data.csv import DatasetInfo
from creditiq_core.models.training import ExperimentResult, TrainingConfig, run_experiment
from creditiq_trainer import io
from creditiq_trainer.config import TrainingFileConfig

ProgressFn = Callable[[float, str], None]


@dataclass
class PreparedData:
    df: pd.DataFrame
    info: DatasetInfo
    mapping: dv.TargetMapping
    predictors: list[str]
    excluded: dict[str, str]
    sensitive: list[str]
    roles: pd.DataFrame
    quality_report: str
    notes: list[str] = field(default_factory=list)


def resolve_path(path: str, base_dirs: list[Path]) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    for base in [Path.cwd(), *base_dirs]:
        candidate = (base / p).resolve()
        if candidate.exists():
            return candidate
    return (Path.cwd() / p).resolve()


def prepare(cfg: TrainingFileConfig, base_dirs: list[Path] | None = None) -> PreparedData:
    path = resolve_path(cfg.dataset.path, base_dirs or [])
    df, info = io.read_csv_path(path, sample_rows=cfg.dataset.load_rows, seed=cfg.seed)
    if cfg.dataset.kind == "synthetic":
        info.is_synthetic = True
        if not any("SYNTHETIC" in n for n in info.notes):
            info.notes.append("SYNTHETIC demonstration data, not real borrowers.")
    definition = cfg.target.definition or csvio.suggest_target_definition(df, cfg.target.column, info.is_synthetic)
    mapping = dv.build_target_mapping(df, cfg.target.column, list(cfg.target.default_values),
                                      None if cfg.target.non_default_values is None
                                      else list(cfg.target.non_default_values), definition)
    summary, profile = dv.profile_dataframe(df)
    issues = dv.find_issues(df, profile)
    roles = dv.suggest_feature_roles(df, cfg.target.column, cfg.columns.date, profile)
    user_excluded = roles["column"].isin(cfg.columns.exclude)
    roles.loc[user_excluded, "use"] = False
    roles.loc[user_excluded, "role"] = "user_excluded"
    roles.loc[user_excluded, "reason"] = "Excluded in the training config"
    sensitive = sorted(set(roles.loc[roles["role"] == "sensitive", "column"]) | set(cfg.columns.sensitive))
    missing = [c for c in [*cfg.columns.sensitive, *cfg.columns.exclude] if c not in df.columns]
    notes = [f"Configured column not in the dataset (ignored): {c}" for c in missing]
    if cfg.columns.keep_sensitive_in_model:
        roles.loc[roles["role"] == "sensitive", "use"] = True
    else:
        is_sensitive = roles["column"].isin(sensitive)
        roles.loc[is_sensitive, "use"] = False
        roles.loc[is_sensitive, "role"] = "sensitive"
    if cfg.columns.exposure and cfg.columns.exposure not in df.columns:
        raise ValueError(f"Exposure column '{cfg.columns.exposure}' is not in the dataset.")
    predictors = roles.loc[roles["use"], "column"].tolist()
    excluded = {str(r.column): str(r.role) for r in roles.itertuples() if not r.use and r.column != cfg.target.column}
    report = dv.quality_report(info.to_dict(), summary, profile, issues, mapping.to_dict(), roles)
    return PreparedData(df, info, mapping, predictors, excluded, [c for c in sensitive if c in df.columns], roles,
                        report, notes)


def core_config(cfg: TrainingFileConfig) -> TrainingConfig:
    return TrainingConfig(
        model_keys=list(cfg.models), sample_size=cfg.sample_size, split_strategy=cfg.split.strategy,
        val_size=cfg.split.validation, test_size=cfg.split.test, seed=cfg.seed, class_weight=cfg.class_weight,
        calibration=cfg.calibration, threshold_rule=cfg.threshold.rule, manual_threshold=cfg.threshold.manual,
        selection_metric=cfg.selection_metric, tune=cfg.tune, use_domain_features=cfg.use_domain_features,
        params=cfg.params, gbm_backend=cfg.gbm_backend, leakage_guard=cfg.leakage_guard.enabled,
        leakage_auc_threshold=cfg.leakage_guard.auc_threshold,
    )


def train(cfg: TrainingFileConfig, data: PreparedData, experiment_id: str | None = None,
          progress: ProgressFn | None = None) -> ExperimentResult:
    """Run the core training algorithm (split → fit on train → calibrate/threshold on validation → test)."""
    return run_experiment(data.df, data.info.to_dict(), data.mapping, data.predictors, core_config(cfg),
                          date_col=cfg.columns.date, exposure_col=cfg.columns.exposure, sensitive_cols=data.sensitive,
                          excluded=data.excluded, progress=progress, experiment_id=experiment_id)
