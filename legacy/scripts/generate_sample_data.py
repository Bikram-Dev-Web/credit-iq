"""Regenerate the bundled SYNTHETIC demo files in data/.

Usage:  python scripts/generate_sample_data.py [--rows 20000] [--seed 42]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config  # noqa: E402
from src.data_loader import generate_synthetic_credit_data  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=20_000)
    parser.add_argument("--batch-rows", type=int, default=5_000)
    parser.add_argument("--seed", type=int, default=config.DEFAULT_SEED)
    args = parser.parse_args()
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    base = generate_synthetic_credit_data(args.rows, args.seed)
    base.to_csv(config.SAMPLE_DATA_FILE, index=False)
    batch = generate_synthetic_credit_data(args.batch_rows, args.seed, shifted=True)
    batch.to_csv(config.SAMPLE_MONITORING_FILE, index=False)
    print(f"Wrote {len(base):,} rows to {config.SAMPLE_DATA_FILE} (default rate {base['TARGET'].mean():.2%})")
    print(f"Wrote {len(batch):,} rows to {config.SAMPLE_MONITORING_FILE} (default rate {batch['TARGET'].mean():.2%})")
    print("Both files are SYNTHETIC demonstration data, not real borrowers.")


if __name__ == "__main__":
    main()
