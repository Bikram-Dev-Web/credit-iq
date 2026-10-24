"""``creditiq-train``: train, register, promote, roll back and export PD models.

Examples::

    creditiq-train train --config services/trainer/configs/synthetic.yaml
    creditiq-train train --config services/trainer/configs/synthetic.yaml --alias production
    creditiq-train versions
    creditiq-train promote --version 2            # staging candidate → production
    creditiq-train rollback                       # production ← previous
    creditiq-train export --ref production --out models/registry
    creditiq-train build-features home-credit --app data/raw/application_train.csv \\
        --bureau data/raw/bureau.csv --previous data/raw/previous_application.csv
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

from creditiq_core.models.training import ExperimentResult
from creditiq_trainer.config import ConfigError, load_config
from creditiq_trainer.pipeline import PreparedData


def _print(obj: object) -> None:
    print(json.dumps(obj, indent=2, default=str))


def cmd_train(args: argparse.Namespace) -> int:
    from creditiq_trainer import file_registry, pipeline

    cfg = load_config(args.config)
    if args.alias is not None:
        cfg.registry.alias = args.alias or None
    if args.no_register:
        cfg.registry.enabled = False
    cfg_path = Path(args.config).resolve()
    data = pipeline.prepare(cfg, base_dirs=[cfg_path.parent, *cfg_path.parents])
    print(f"Loaded {len(data.df):,} rows × {data.df.shape[1]} columns from {data.info.name}"
          + (" (SYNTHETIC)" if data.info.is_synthetic else ""))
    m = data.mapping
    print(f"Target {m.column}: {m.n_positive:,} defaults / {m.n_negative:,} non-defaults ({m.prevalence:.2%}); "
          f"{m.n_unmapped:,} unmapped rows excluded")
    for note in [*m.warnings, *data.notes]:
        print("  note:", note)
    print(f"{len(data.predictors)} predictors; sensitive (audit only): {', '.join(data.sensitive) or 'none'}")

    def progress(f: float, msg: str) -> None:
        print(f"  [{f:4.0%}] {msg}")

    if args.no_mlflow:
        result = pipeline.train(cfg, data, progress=progress)
        from creditiq_trainer.package_writer import write_package

        with tempfile.TemporaryDirectory() as tmp:
            package = write_package(cfg, data, result, Path(tmp) / "package")
            run: dict[str, object] = {"run_id": None, "package": None}
            if args.file_registry:
                aliases = [cfg.registry.alias] if cfg.registry.alias else []
                version = file_registry.add_version(Path(args.file_registry), cfg.registry.model_name, package,
                                                    aliases=aliases, info=_version_info(result, data))
                run = {"run_id": None, "file_registry_version": version, "aliases": aliases}
    else:
        import mlflow

        from creditiq_trainer import tracking

        uri = tracking.configure(args.tracking_uri, args.experiment)
        with mlflow.start_run(run_name=cfg.name) as active:
            result = pipeline.train(cfg, data, experiment_id=active.info.run_id, progress=progress)
            tr = tracking.log_run(cfg, data, result, active.info.run_id)
        run = {"tracking_uri": uri, "run_id": tr.run_id, "model": tr.model_name, "version": tr.version,
               "alias": tr.alias}
        if args.file_registry and tr.version:
            file_registry.add_version(Path(args.file_registry), cfg.registry.model_name, tr.package_dir,
                                      version=tr.version, aliases=[tr.alias] if tr.alias else [],
                                      info=_version_info(result, data))
    for w in result.warnings:
        print("  warning:", w)
    comp = result.comparison_table()
    cols = ["model", "selected", "val_roc_auc", "test_roc_auc", "test_pr_auc", "test_brier", "test_ks", "threshold"]
    print("\nHeld-out results" + (" (SYNTHETIC data)" if data.info.is_synthetic else "") + ":")
    print(comp[cols].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    _print(run)
    return 0


def _version_info(result: ExperimentResult, data: PreparedData) -> dict[str, object]:
    test = result.metrics[result.selected]["test"]
    return {"selected_model": result.selected, "label": result.selected_model.label,
            "test_roc_auc": test["roc_auc"], "test_pr_auc": test["pr_auc"], "test_brier": test["brier"],
            "is_synthetic": data.info.is_synthetic, "dataset": data.info.name, "run_id": result.experiment_id}


def cmd_versions(args: argparse.Namespace) -> int:
    from creditiq_trainer import tracking

    tracking.configure(args.tracking_uri, args.experiment)
    _print(tracking.list_versions(args.model))
    return 0


def cmd_promote(args: argparse.Namespace) -> int:
    from creditiq_trainer import tracking

    tracking.configure(args.tracking_uri, args.experiment)
    _print(tracking.promote(args.model, str(args.version), args.alias))
    return 0


def cmd_rollback(args: argparse.Namespace) -> int:
    from creditiq_trainer import tracking

    tracking.configure(args.tracking_uri, args.experiment)
    _print(tracking.rollback(args.model))
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    """Copy registered versions (with their aliases) from MLflow into a file registry."""
    from creditiq_trainer import file_registry, tracking

    tracking.configure(args.tracking_uri, args.experiment)
    versions = tracking.list_versions(args.model)
    wanted = [v for v in versions if args.ref in ("all", v["version"], *v["aliases"])]
    if not wanted:
        print(f"No version of {args.model} matches '{args.ref}'.", file=sys.stderr)
        return 1
    for v in wanted:
        with tempfile.TemporaryDirectory() as tmp:
            package = tracking.download_package(args.model, v["version"], Path(tmp))
            meta = json.loads((package / "metadata.json").read_text(encoding="utf-8"))
            sel = meta.get("artifact_model") or meta.get("selected_model")
            test = meta.get("metrics", {}).get(sel, {}).get("test", {})
            info = {"selected_model": sel, "run_id": v["run_id"], "test_roc_auc": test.get("roc_auc"),
                    "test_pr_auc": test.get("pr_auc"), "test_brier": test.get("brier"),
                    "is_synthetic": v["tags"].get("creditiq.is_synthetic") == "true",
                    "dataset": v["tags"].get("creditiq.dataset_name"), "description": v["description"]}
            file_registry.add_version(Path(args.out), args.model, package, version=v["version"],
                                      aliases=v["aliases"], info=info)
        print(f"exported {args.model} v{v['version']} {v['aliases']} → {args.out}")
    return 0


def cmd_build_features(args: argparse.Namespace) -> int:
    from creditiq_trainer.datasets import home_credit

    home_credit.build(args.app, args.bureau, args.previous, args.out)
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="creditiq-train", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--tracking-uri", default=None, help="MLflow tracking URI (default: $MLFLOW_TRACKING_URI or "
                                                            "sqlite:///mlruns/mlflow.db)")
    common.add_argument("--experiment", default=None, help="MLflow experiment name (default: creditiq)")
    common.add_argument("--model", default="creditiq-pd", help="Registered model name")
    sub = ap.add_subparsers(dest="command", required=True)

    t = sub.add_parser("train", parents=[common], help="Train from a YAML config, log to MLflow and register")
    t.add_argument("--config", required=True)
    t.add_argument("--alias", default=None, help="Alias for the new version (overrides the config; '' for none)")
    t.add_argument("--no-register", action="store_true", help="Log the run but do not register a model version")
    t.add_argument("--no-mlflow", action="store_true", help="Train without MLflow (use with --file-registry)")
    t.add_argument("--file-registry", default=None, help="Also write the version into this file registry")
    t.set_defaults(func=cmd_train)

    v = sub.add_parser("versions", parents=[common], help="List registered versions and aliases")
    v.set_defaults(func=cmd_versions)
    p = sub.add_parser("promote", parents=[common], help="Point an alias (default: production) at a version")
    p.add_argument("--version", required=True)
    p.add_argument("--alias", default="production")
    p.set_defaults(func=cmd_promote)
    r = sub.add_parser("rollback", parents=[common], help="Move production back to the previous version")
    r.set_defaults(func=cmd_rollback)
    e = sub.add_parser("export", parents=[common], help="Export registered versions to a file registry")
    e.add_argument("--ref", default="production", help="Version number, alias, or 'all'")
    e.add_argument("--out", required=True)
    e.set_defaults(func=cmd_export)

    b = sub.add_parser("build-features", help="Dataset enrichment")
    bsub = b.add_subparsers(dest="dataset", required=True)
    hc = bsub.add_parser("home-credit", help="Join bureau and previous-application aggregates")
    hc.add_argument("--app", required=True, type=Path)
    hc.add_argument("--bureau", type=Path)
    hc.add_argument("--previous", type=Path)
    hc.add_argument("--out", type=Path, default=Path("data/raw/home_credit_enriched.csv"))
    hc.set_defaults(func=cmd_build_features)
    return ap


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):  # Windows consoles default to a legacy code page
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = build_parser().parse_args(argv)
    try:
        code: int = args.func(args)
        return code
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
