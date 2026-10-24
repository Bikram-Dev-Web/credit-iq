"""Convert UCI Statlog German Credit ``german.data`` into a CSV with named columns.

Usage:
    python -m creditiq_trainer.datasets.german data/raw/german.data data/raw/german_credit.csv

The raw file is space-separated with no header: 20 attributes followed by the
class label (1 = good, 2 = bad). The label is kept exactly as published — in the
training config you must map 2 → default yourself. ``personal_status_sex`` and
``age_years`` (and arguably ``foreign_worker``) are sensitive attributes and are
excluded from the model by default; they remain available for the fairness audit.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

COLUMNS = [
    "checking_account_status", "duration_months", "credit_history", "purpose", "credit_amount",
    "savings_account", "employment_since", "installment_rate_pct_income", "personal_status_sex",
    "other_debtors_guarantors", "residence_since", "property", "age_years", "other_installment_plans",
    "housing", "existing_credits_at_bank", "job", "people_liable", "telephone", "foreign_worker",
    "credit_risk_class",
]


def convert(src: Path, dst: Path) -> pd.DataFrame:
    df = pd.read_csv(src, sep=r"\s+", header=None, engine="python")
    if df.shape[1] != len(COLUMNS):
        raise SystemExit(f"Expected {len(COLUMNS)} fields per line, found {df.shape[1]}. Is this german.data (not german.data-numeric)?")
    df.columns = COLUMNS
    dst.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(dst, index=False)
    print(f"Wrote {len(df):,} rows to {dst}. Class counts: {df['credit_risk_class'].value_counts().to_dict()} (1 = good, 2 = bad)")
    return df


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    convert(Path(sys.argv[1]), Path(sys.argv[2]))
