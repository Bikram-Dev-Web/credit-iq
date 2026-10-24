"""Train and evaluate CreditIQ models from the command line (no UI needed).

Examples
--------
Bundled synthetic sample:
    python scripts/train_cli.py --data data/sample_credit_data.csv --target TARGET --default-label 1 \
        --date-col APPLICATION_DATE --exposure-col AMT_CREDIT

Home Credit application_train.csv (after downloading it into data/raw/):
    python scripts/train_cli.py --data data/raw/application_train.csv --target TARGET --default-label 1 \
        --exposure-col AMT_CREDIT --load-rows 150000 --sample 60000

The default label is a required argument on purpose: CreditIQ never assumes
that 1 means default.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config  # noqa: E402
from src import data_loader as dl  # noqa: E402
from src import data_validation as dv  # noqa: E402
from src import modeling as mdl  # noqa: E402
from src import persistence as ps  # noqa: E402
from src import reporting as rp  # noqa: E402


def _label(value: str):
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="CSV file")
    ap.add_argument("--target", required=True, help="Outcome column")
    ap.add_argument("--default-label", required=True, nargs="+", help="Label value(s) meaning default")
    ap.add_argument("--non-default-label", nargs="*", help="Label value(s) meaning non-default (default: all others)")
    ap.add_argument("--date-col", default=None)
    ap.add_argument("--exposure-col", default=None)
    ap.add_argument("--models", default="logreg,random_forest,gbm")
    ap.add_argument("--split", default="stratified", choices=["stratified", "chronological"])
    ap.add_argument("--load-rows", type=int, default=None, help="Randomly sample this many rows while reading")
    ap.add_argument("--sample", type=int, default=config.DEFAULT_TRAIN_SAMPLE, help="Max labelled rows used for training")
    ap.add_argument("--calibration", default="sigmoid", choices=["sigmoid", "isotonic", "none"])
    ap.add_argument("--tune", action="store_true")
    ap.add_argument("--seed", type=int, default=config.DEFAULT_SEED)
    ap.add_argument("--keep-sensitive", action="store_true", help="Do not auto-exclude potentially protected attributes")
    ap.add_argument("--save-name", default=None, help="Save the selected model bundle to models/<name>.joblib")
    args = ap.parse_args()

    config.ensure_dirs()
    df, info = dl.read_csv_file(Path(args.data), sample_rows=args.load_rows, seed=args.seed, source_kind="local")
    if Path(args.data).resolve() == config.SAMPLE_DATA_FILE.resolve():
        info.is_synthetic = True
        info.notes.append("SYNTHETIC demonstration data.")
    print(f"Loaded {len(df):,} rows × {df.shape[1]} columns from {info.name}")

    positives = [_label(v) for v in args.default_label]
    negatives = [_label(v) for v in args.non_default_label] if args.non_default_label else None
    mapping = dv.build_target_mapping(df, args.target, positives, negatives,
                                      dl.suggest_target_definition(df, args.target, info.is_synthetic))
    print(f"Target {args.target}: {mapping.n_positive:,} defaults / {mapping.n_negative:,} non-defaults "
          f"({mapping.prevalence:.2%}); {mapping.n_unmapped:,} unmapped rows excluded")
    for w in mapping.warnings:
        print("  warning:", w)

    roles = dv.suggest_feature_roles(df, args.target, args.date_col)
    if args.keep_sensitive:
        roles.loc[roles["role"] == "sensitive", "use"] = True
    # No full-data leakage scan here: the training pipeline's leakage guard screens candidates on the TRAINING split
    # only (so held-out rows never influence feature selection) and reports what it removed.
    predictors = roles.loc[roles["use"], "column"].tolist()
    excluded = {r.column: r.role for r in roles.itertuples() if not r.use and r.column != args.target}
    print(f"{len(predictors)} predictors; excluded: {excluded}")

    cfg = mdl.TrainingConfig(model_keys=args.models.split(","), sample_size=args.sample, split_strategy=args.split,
                             seed=args.seed, calibration=args.calibration, tune=args.tune)
    result = mdl.run_experiment(df, info.to_dict(), mapping, predictors, cfg, date_col=args.date_col,
                                exposure_col=args.exposure_col,
                                sensitive_cols=roles.loc[roles["role"] == "sensitive", "column"].tolist(),
                                excluded=excluded, progress=lambda f, m: print(f"  [{f:4.0%}] {m}"))
    guard = {c: r for c, r in result.excluded.items() if r.startswith("leakage (training-split")}
    if guard:
        print("Excluded by the training-split leakage guard:", ", ".join(guard))
    for w in result.warnings:
        print("  warning:", w)
    comp = result.comparison_table()
    cols = ["model", "selected", "val_roc_auc", "test_roc_auc", "test_pr_auc", "test_brier", "threshold", "test_recall", "test_precision"]
    print("\nHeld-out results" + (" (SYNTHETIC data)" if info.is_synthetic else "") + ":")
    print(comp[cols].to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    ps.save_experiment(result)
    if args.save_name:
        path = ps.save_bundle(result.bundle(), args.save_name)
        ps.update_experiment_artifact(result.experiment_id, str(path))
        print(f"\nSaved model bundle: {path}")
    report = ps.save_report(rp.to_markdown(rp.build_report({"dataset": info.to_dict(), "target": mapping.to_dict(),
                                                             "experiment": result})),
                            f"cli_report_{result.experiment_id}.md")
    print(f"Report: {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
