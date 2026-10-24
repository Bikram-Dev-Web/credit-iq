"""Reading datasets from disk."""
import io as stdio
from pathlib import Path

import pandas as pd
import pytest

from creditiq_core.data import synthetic as synth
from creditiq_trainer import io

DATA_DIR = Path(__file__).resolve().parents[3] / "data"


def test_read_local_csv_chunked_sampling(tmp_path):
    p = tmp_path / "big.csv"
    pd.DataFrame({"a": range(5000)}).to_csv(p, index=False)
    out, info = io.read_csv_path(p, sample_rows=500)
    assert info.sampled and 300 < len(out) < 700  # chunk-wise Bernoulli sampling ≈ 500 rows
    assert info.n_rows_file == 5000 and info.source == "local"


def test_read_local_csv_without_sampling_and_bad_paths(tmp_path):
    p = tmp_path / "semi.csv"
    p.write_bytes(b"a;b\n1;2\n3;4\n")
    out, info = io.read_csv_path(p)
    assert list(out.columns) == ["a", "b"] and not info.sampled and info.n_rows_file == 2
    with pytest.raises(FileNotFoundError):
        io.read_csv_path(tmp_path / "missing.csv")
    (tmp_path / "x.txt").write_text("a\n1\n")
    with pytest.raises(ValueError, match="csv"):
        io.read_csv_path(tmp_path / "x.txt")


def test_bundled_synthetic_files_are_loaded_and_labelled():
    df, info = io.load_synthetic_dataset(DATA_DIR)
    assert len(df) == 20_000 and info.is_synthetic
    batch = io.load_synthetic_monitoring_batch(DATA_DIR)
    assert len(batch) == 5_000 and batch["SK_ID_CURR"].min() >= 900_000


def test_bundled_sample_matches_the_generator():
    """The committed CSV is exactly what the seeded generator produces, so it can always be rebuilt."""
    bundled = pd.read_csv(DATA_DIR / io.SAMPLE_DATA_FILE, low_memory=False)
    fresh = synth.generate_synthetic_credit_data(20_000, seed=42)
    fresh = pd.read_csv(stdio.StringIO(fresh.to_csv(index=False)), low_memory=False)  # same CSV round trip
    pd.testing.assert_frame_equal(bundled, fresh, check_exact=False, rtol=1e-9)
