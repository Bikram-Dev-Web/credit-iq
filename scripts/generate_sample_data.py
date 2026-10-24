"""Regenerate the bundled SYNTHETIC demo files in data/.

Usage:  uv run python scripts/generate_sample_data.py [--rows 20000] [--batch-rows 5000] [--seed 42]
"""
from __future__ import annotations

import argparse
from pathlib import Path

from creditiq_core import defaults
from creditiq_core.data.synthetic import generate_synthetic_credit_data

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=20_000)
    parser.add_argument("--batch-rows", type=int, default=5_000)
    parser.add_argument("--seed", type=int, default=defaults.DEFAULT_SEED)
    parser.add_argument("--out-dir", type=Path, default=DATA_DIR)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    base = generate_synthetic_credit_data(args.rows, args.seed)
    base.to_csv(args.out_dir / "sample_credit_data.csv", index=False)
    batch = generate_synthetic_credit_data(args.batch_rows, args.seed, shifted=True)
    batch.to_csv(args.out_dir / "sample_monitoring_batch.csv", index=False)
    print(f"Wrote {len(base):,} rows (default rate {base['TARGET'].mean():.2%}) and a shifted batch of "
          f"{len(batch):,} rows (default rate {batch['TARGET'].mean():.2%}) to {args.out_dir}")
    print("Both files are SYNTHETIC demonstration data, not real borrowers.")


if __name__ == "__main__":
    main()
