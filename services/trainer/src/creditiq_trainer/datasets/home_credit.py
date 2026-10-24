"""Optionally enrich Home Credit application_train.csv with bureau and previous-application history.

Usage (after downloading the Kaggle files into data/raw/):

    python -m creditiq_trainer.datasets.home_credit \
        --app data/raw/application_train.csv \
        --bureau data/raw/bureau.csv \
        --previous data/raw/previous_application.csv \
        --out data/raw/home_credit_enriched.csv

Both related tables describe credit history that, according to the dataset's
documentation, pre-dates the current application (``DAYS_*`` columns are
relative to the application date), so per-applicant aggregates are available at
application time. They are streamed in chunks and reduced to one row per
``SK_ID_CURR`` so the large files are never loaded into memory at once.

Only columns listed below are read. If your copy of the files differs, the
script stops with a clear message rather than guessing.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

BUREAU_COLS = ["SK_ID_CURR", "CREDIT_ACTIVE", "DAYS_CREDIT", "CREDIT_DAY_OVERDUE", "AMT_CREDIT_SUM",
               "AMT_CREDIT_SUM_DEBT", "AMT_CREDIT_SUM_OVERDUE", "CNT_CREDIT_PROLONG"]
PREV_COLS = ["SK_ID_CURR", "NAME_CONTRACT_STATUS", "AMT_APPLICATION", "AMT_CREDIT", "DAYS_DECISION"]


def _check_columns(path: Path, needed: list[str]) -> None:
    header = pd.read_csv(path, nrows=0).columns
    missing = [c for c in needed if c not in header]
    if missing:
        raise SystemExit(f"{path.name} is missing expected columns: {missing}")


def aggregate_bureau(path: Path, chunksize: int = 200_000) -> pd.DataFrame:
    _check_columns(path, BUREAU_COLS)
    parts = []
    for chunk in pd.read_csv(path, usecols=BUREAU_COLS, chunksize=chunksize):
        chunk["is_active"] = (chunk["CREDIT_ACTIVE"] == "Active").astype(int)
        g = chunk.groupby("SK_ID_CURR")
        parts.append(pd.DataFrame({
            "n": g.size(), "active": g["is_active"].sum(), "days_credit_sum": g["DAYS_CREDIT"].sum(),
            "days_credit_min": g["DAYS_CREDIT"].min(), "overdue_days_max": g["CREDIT_DAY_OVERDUE"].max(),
            "credit_sum": g["AMT_CREDIT_SUM"].sum(min_count=1), "debt_sum": g["AMT_CREDIT_SUM_DEBT"].sum(min_count=1),
            "overdue_sum": g["AMT_CREDIT_SUM_OVERDUE"].sum(min_count=1), "prolong_sum": g["CNT_CREDIT_PROLONG"].sum(),
        }))
    a = pd.concat(parts).groupby(level=0).agg({"n": "sum", "active": "sum", "days_credit_sum": "sum",
                                               "days_credit_min": "min", "overdue_days_max": "max", "credit_sum": "sum",
                                               "debt_sum": "sum", "overdue_sum": "sum", "prolong_sum": "sum"})
    out = pd.DataFrame(index=a.index)
    out["BUREAU_LOAN_COUNT"] = a["n"]
    out["BUREAU_ACTIVE_COUNT"] = a["active"]
    out["BUREAU_DAYS_CREDIT_MEAN"] = a["days_credit_sum"] / a["n"]
    out["BUREAU_DAYS_CREDIT_MIN"] = a["days_credit_min"]
    out["BUREAU_MAX_DAYS_OVERDUE"] = a["overdue_days_max"]
    out["BUREAU_CREDIT_SUM"] = a["credit_sum"]
    out["BUREAU_DEBT_SUM"] = a["debt_sum"]
    out["BUREAU_DEBT_TO_CREDIT"] = (a["debt_sum"] / a["credit_sum"].replace(0, np.nan))
    out["BUREAU_OVERDUE_SUM"] = a["overdue_sum"]
    out["BUREAU_PROLONG_COUNT"] = a["prolong_sum"]
    return out


def aggregate_previous(path: Path, chunksize: int = 200_000) -> pd.DataFrame:
    _check_columns(path, PREV_COLS)
    parts = []
    for chunk in pd.read_csv(path, usecols=PREV_COLS, chunksize=chunksize):
        chunk["approved"] = (chunk["NAME_CONTRACT_STATUS"] == "Approved").astype(int)
        chunk["refused"] = (chunk["NAME_CONTRACT_STATUS"] == "Refused").astype(int)
        g = chunk.groupby("SK_ID_CURR")
        parts.append(pd.DataFrame({"n": g.size(), "approved": g["approved"].sum(), "refused": g["refused"].sum(),
                                   "amt_app_sum": g["AMT_APPLICATION"].sum(min_count=1), "amt_app_n": g["AMT_APPLICATION"].count(),
                                   "days_decision_max": g["DAYS_DECISION"].max()}))
    a = pd.concat(parts).groupby(level=0).agg({"n": "sum", "approved": "sum", "refused": "sum", "amt_app_sum": "sum",
                                               "amt_app_n": "sum", "days_decision_max": "max"})
    out = pd.DataFrame(index=a.index)
    out["PREV_APP_COUNT"] = a["n"]
    out["PREV_APPROVED_COUNT"] = a["approved"]
    out["PREV_REFUSED_COUNT"] = a["refused"]
    out["PREV_REFUSED_SHARE"] = a["refused"] / a["n"]
    out["PREV_AMT_APPLICATION_MEAN"] = a["amt_app_sum"] / a["amt_app_n"].replace(0, np.nan)
    out["PREV_DAYS_DECISION_LATEST"] = a["days_decision_max"]
    return out


def build(app: Path, bureau: Path | None, previous: Path | None, out: Path) -> pd.DataFrame:
    df = pd.read_csv(app, low_memory=False)
    if "SK_ID_CURR" not in df.columns:
        raise SystemExit("The application file has no SK_ID_CURR column.")
    for path, fn, label in ((bureau, aggregate_bureau, "bureau"), (previous, aggregate_previous, "previous applications")):
        if path is None:
            continue
        agg = fn(path)
        df = df.merge(agg, how="left", left_on="SK_ID_CURR", right_index=True)
        print(f"Added {agg.shape[1]} {label} features for {agg.shape[0]:,} applicants")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Wrote {len(df):,} rows × {df.shape[1]} columns to {out}")
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--app", required=True, type=Path)
    ap.add_argument("--bureau", type=Path)
    ap.add_argument("--previous", type=Path)
    ap.add_argument("--out", type=Path, default=Path("data/raw/home_credit_enriched.csv"))
    args = ap.parse_args()
    build(args.app, args.bureau, args.previous, args.out)


if __name__ == "__main__":
    main()
