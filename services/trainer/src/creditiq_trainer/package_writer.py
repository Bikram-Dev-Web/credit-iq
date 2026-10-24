"""Write a model-version package directory (layout defined in ``creditiq_core.models.package``)."""
from __future__ import annotations

import json
import platform
import subprocess
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml

import creditiq_core
from creditiq_core import evaluation as ev
from creditiq_core.jsonsafe import to_json
from creditiq_core.models import package as pkg
from creditiq_core.models.training import ExperimentResult
from creditiq_core.reporting.model_card import model_card_markdown
from creditiq_trainer.config import TrainingFileConfig
from creditiq_trainer.pipeline import PreparedData


def git_sha() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=5,
                             check=False)
        return out.stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def run_tags(cfg: TrainingFileConfig, data: PreparedData, result: ExperimentResult) -> dict[str, str]:
    """Lineage recorded on every run and registered version."""
    return {
        "creditiq.config_name": cfg.name,
        "creditiq.config_hash": cfg.config_hash(),
        "creditiq.dataset_name": data.info.name,
        "creditiq.dataset_fingerprint": data.info.fingerprint,
        "creditiq.is_synthetic": str(data.info.is_synthetic).lower(),
        "creditiq.selected_model": result.selected,
        "creditiq.git_sha": git_sha(),
        "creditiq.core_version": creditiq_core.__version__,
        "creditiq.seed": str(cfg.seed),
        "creditiq.python": platform.python_version(),
    }


def _plot_curves(result: ExperimentResult, out: Path) -> list[Path]:
    """Calibration, ROC and PR plots for every model on the test split (PNG, logged to MLflow)."""
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    y = result.y_split("test")
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], ls="--", lw=1, color="grey", label="perfect calibration")
    for key, model in result.models.items():
        cal = ev.calibration_table(y, result.pd_split("test", key))
        ax.plot(cal["mean_pd"], cal["observed_rate"], marker="o", label=model.label)
    ax.set(xlabel="Mean predicted PD", ylabel="Observed default rate", title="Calibration (test split)")
    ax.legend(fontsize=7)
    paths.append(out / "calibration_test.png")
    fig.savefig(paths[-1], dpi=120, bbox_inches="tight")
    plt.close(fig)
    for kind, xcol, ycol, title in (("roc", "fpr", "tpr", "ROC (test split)"),
                                    ("pr", "recall", "precision", "Precision–recall (test split)")):
        fig, ax = plt.subplots(figsize=(5, 5))
        for key, model in result.models.items():
            pts = (ev.roc_points if kind == "roc" else ev.pr_points)(y, result.pd_split("test", key))
            ax.plot(pts[xcol], pts[ycol], label=model.label)
        if kind == "roc":
            ax.plot([0, 1], [0, 1], ls="--", lw=1, color="grey")
        else:
            ax.axhline(y.mean(), ls="--", lw=1, color="grey", label="default rate (random baseline)")
        ax.set(xlabel=xcol.upper() if kind == "roc" else "Recall", ylabel=ycol.upper() if kind == "roc" else "Precision",
               title=title)
        ax.legend(fontsize=7)
        paths.append(out / f"{kind}_test.png")
        fig.savefig(paths[-1], dpi=120, bbox_inches="tight")
        plt.close(fig)
    return paths


def write_package(cfg: TrainingFileConfig, data: PreparedData, result: ExperimentResult, out_dir: Path,
                  extra_metadata: dict[str, Any] | None = None) -> Path:
    """Write the selected model's package. Returns the directory."""
    out_dir.mkdir(parents=True, exist_ok=True)
    artifact = result.to_artifact()
    artifact.metadata["lineage"] = {**run_tags(cfg, data, result), **(extra_metadata or {})}
    (out_dir / pkg.ARTIFACT_FILE).write_bytes(pkg.artifact_to_bytes(artifact))
    snapshot = result.evaluation_snapshot()
    snapshot.frame.to_parquet(out_dir / pkg.SNAPSHOT_FILE, index=False)
    (out_dir / pkg.SNAPSHOT_META_FILE).write_text(to_json(pkg.snapshot_meta(snapshot), indent=2), encoding="utf-8")
    (out_dir / pkg.EVALUATION_FILE).write_text(to_json(pkg.evaluation_report(result)), encoding="utf-8")
    (out_dir / pkg.METADATA_FILE).write_text(to_json(artifact.metadata, indent=2), encoding="utf-8")
    (out_dir / pkg.MODEL_CARD_FILE).write_text(model_card_markdown(artifact.metadata, result.comparison_table()),
                                               encoding="utf-8")
    (out_dir / pkg.QUALITY_REPORT_FILE).write_text(data.quality_report, encoding="utf-8")
    (out_dir / "config.yaml").write_text(yaml.safe_dump(cfg.model_dump(mode="json"), sort_keys=False),
                                         encoding="utf-8")
    _plot_curves(result, out_dir / "plots")
    return out_dir


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))
