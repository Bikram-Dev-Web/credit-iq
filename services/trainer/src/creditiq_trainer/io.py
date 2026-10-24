"""Reading datasets from disk. Parsing and validation are delegated to ``creditiq_core.data.csv``."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from creditiq_core import defaults
from creditiq_core.data.csv import DatasetInfo, finalize_frame, read_csv_file, sniff
from creditiq_core.data.synthetic import generate_synthetic_credit_data, synthetic_info

SAMPLE_DATA_FILE = "sample_credit_data.csv"
SAMPLE_MONITORING_FILE = "sample_monitoring_batch.csv"


def _count_data_rows(path: Path) -> int:
    """Fast newline count (minus header) without parsing the CSV."""
    count = 0
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            count += chunk.count(b"\n")
    return max(count - 1, 0)


def read_csv_path(
    path: str | Path,
    *,
    sample_rows: int | None = None,
    seed: int = defaults.DEFAULT_SEED,
    optimize_memory: bool = True,
) -> tuple[pd.DataFrame, DatasetInfo]:
    """Read a CSV from disk. With ``sample_rows``, the file is streamed in 100k-row chunks and each chunk is
    sampled at the same rate, so a large file (e.g. Home Credit's ~300k rows) is never held in memory at once."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    if path.suffix.lower() != ".csv":
        raise ValueError("Only .csv files are supported.")
    with open(path, "rb") as fh:
        sep, encoding = sniff(fh.read(65_536))
    n_rows_file = _count_data_rows(path)
    if not (sample_rows and n_rows_file > sample_rows):
        with open(path, "rb") as fh:
            df, info = read_csv_file(fh, name=path.name, optimize_memory=optimize_memory, source_kind="local")
        info.n_rows_file = n_rows_file
        return df, info
    frac = sample_rows / n_rows_file
    rng = np.random.default_rng(seed)
    parts = []
    for chunk in pd.read_csv(path, low_memory=False, chunksize=100_000, sep=sep, encoding=encoding):
        parts.append(chunk.loc[rng.random(len(chunk)) < frac])
    df = pd.concat(parts, ignore_index=True)
    notes = [f"Random sample of ~{frac:.1%} of {n_rows_file:,} rows (seed {seed})."]
    return finalize_frame(df, name=path.name, sep=sep, source_kind="local", n_rows_file=n_rows_file, sampled=True,
                          notes=notes, optimize=optimize_memory)


def load_synthetic_dataset(data_dir: str | Path, n_rows: int = 20_000,
                           seed: int = defaults.DEFAULT_SEED) -> tuple[pd.DataFrame, DatasetInfo]:
    """The bundled synthetic CSV when it matches the request, otherwise a freshly generated copy."""
    path = Path(data_dir) / SAMPLE_DATA_FILE
    if path.exists() and n_rows == 20_000 and seed == defaults.DEFAULT_SEED:
        df = pd.read_csv(path, low_memory=False)
    else:
        df = generate_synthetic_credit_data(n_rows=n_rows, seed=seed)
    return df, synthetic_info(df, seed)


def load_synthetic_monitoring_batch(data_dir: str | Path, n_rows: int = 5_000,
                                    seed: int = defaults.DEFAULT_SEED) -> pd.DataFrame:
    """The shifted synthetic batch used to demonstrate drift monitoring."""
    path = Path(data_dir) / SAMPLE_MONITORING_FILE
    if path.exists() and n_rows == 5_000 and seed == defaults.DEFAULT_SEED:
        return pd.read_csv(path, low_memory=False)
    return generate_synthetic_credit_data(n_rows=n_rows, seed=seed, shifted=True)
