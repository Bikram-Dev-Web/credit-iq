"""CSV parsing from bytes or file-like objects, plus the dataset provenance record.

The core never opens paths: services read files (or HTTP uploads) and hand the content
here. Separator and encoding are sniffed, and binary or header-only files are rejected
with a readable message.
"""
from __future__ import annotations

import hashlib
import io
import re
from dataclasses import dataclass, field
from pathlib import PurePath
from typing import IO, Any

import numpy as np
import pandas as pd

from creditiq_core import defaults

SYNTHETIC_TARGET_DEFINITION = (
    "SYNTHETIC: TARGET = 1 marks a simulated 'payment difficulty' event generated "
    "by a known random process, modelled on the Home Credit target definition. "
    "It is not an observation of any real borrower."
)

HOME_CREDIT_TARGET_DEFINITION = (
    "Home Credit TARGET (per the dataset's column description file): 1 = client with "
    "payment difficulties — a late payment of more than X days on at least one of the "
    "first Y instalments of the loan in the sample; 0 = all other cases. X and Y are not "
    "disclosed, so this is a payment-difficulty event, not a universal legal or "
    "regulatory definition of default."
)

SYNTHETIC_SIGNATURE = {"SK_ID_CURR", "TARGET", "APPLICATION_DATE", "POST_LOAN_COLLECTION_CALLS"}
SYNTHETIC_FILE_NAMES = {"sample_credit_data.csv", "sample_monitoring_batch.csv"}


@dataclass
class DatasetInfo:
    """Provenance record for a loaded dataset."""

    name: str
    source: str  # "synthetic" | "upload" | "local"
    is_synthetic: bool
    n_rows_loaded: int
    n_cols: int
    n_rows_file: int | None = None
    sampled: bool = False
    fingerprint: str = ""
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "source": self.source,
            "is_synthetic": self.is_synthetic,
            "n_rows_loaded": self.n_rows_loaded,
            "n_cols": self.n_cols,
            "n_rows_file": self.n_rows_file,
            "sampled": self.sampled,
            "fingerprint": self.fingerprint,
            "notes": list(self.notes),
        }


def sanitize_filename(name: str, default: str = "file", max_len: int = 80) -> str:
    """Return a filesystem-safe file name (no paths, no leading dots)."""
    base = PurePath(str(name).replace("\\", "/")).name  # strip any directory components
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", base).strip("._")
    return (cleaned or default)[:max_len]


def dataframe_fingerprint(df: pd.DataFrame, name: str = "") -> str:
    """Stable short hash used to key caches and experiment records."""
    h = hashlib.sha1()
    h.update(name.encode())
    h.update(str(df.shape).encode())
    h.update("|".join(map(str, df.columns)).encode())
    if len(df):
        sample = df.head(2000)
        try:
            h.update(pd.util.hash_pandas_object(sample, index=False).to_numpy().tobytes())
        except TypeError:  # unhashable cells (rare) – fall back to string form
            h.update(sample.astype(str).to_csv(index=False).encode())
    return h.hexdigest()[:12]


def optimize_memory(df: pd.DataFrame) -> pd.DataFrame:
    """Downcast numeric columns to save memory on large files."""
    for col in df.columns:
        s = df[col]
        if pd.api.types.is_float_dtype(s):
            df[col] = pd.to_numeric(s, downcast="float")
        elif pd.api.types.is_integer_dtype(s) and not pd.api.types.is_bool_dtype(s):
            df[col] = pd.to_numeric(s, downcast="integer")
    return df


def is_bundled_synthetic(df: pd.DataFrame, name: str | None) -> bool:
    """True for the bundled synthetic files, however they are loaded (local file, upload, renamed copy)."""
    return (name or "").lower() in SYNTHETIC_FILE_NAMES or SYNTHETIC_SIGNATURE.issubset(df.columns)


def sniff(head: bytes) -> tuple[str, str]:
    """Return (separator, encoding) from the first bytes of a CSV."""
    if b"\x00" in head:
        raise ValueError("This does not look like a text CSV file (it contains binary data). "
                         "Export the table as CSV (UTF-8) and try again.")
    cut = head.rsplit(b"\n", 1)[0] if b"\n" in head else head
    try:
        text, encoding = cut.decode("utf-8-sig"), "utf-8-sig"
    except UnicodeDecodeError:
        text, encoding = cut.decode("latin-1"), "latin-1"
    lines = [ln for ln in text.splitlines()[:20] if ln.strip()]
    best, best_score = ",", 0.0
    for sep in (",", ";", "\t", "|"):
        counts = [ln.count(sep) for ln in lines]
        if counts and counts[0] > 0:
            consistent = sum(c == counts[0] for c in counts) / len(counts)
            score = counts[0] * consistent
            if score > best_score:
                best, best_score = sep, score
    return best, encoding


def _head_bytes(source: IO[bytes], size: int = 65_536) -> bytes:
    pos = source.tell()
    head = source.read(size)
    source.seek(pos)
    return head


def _read(source: IO[bytes], sep: str, encoding: str) -> pd.DataFrame:
    try:
        return pd.read_csv(source, low_memory=False, sep=sep, encoding=encoding)
    except UnicodeDecodeError:
        source.seek(0)
        return pd.read_csv(source, low_memory=False, sep=sep, encoding="latin-1")


def finalize_frame(
    df: pd.DataFrame,
    *,
    name: str,
    sep: str,
    source_kind: str,
    n_rows_file: int | None,
    sampled: bool,
    notes: list[str],
    optimize: bool = True,
) -> tuple[pd.DataFrame, DatasetInfo]:
    """Validate a freshly parsed table and build its provenance record (shared by every reader)."""
    if df.shape[1] == 0:
        raise ValueError("The CSV has no columns.")
    if len(df) == 0:
        raise ValueError("The CSV has a header but no data rows.")
    df.columns = pd.Index([str(c).strip() for c in df.columns])
    if df.columns.duplicated().any():
        dupes = df.columns[df.columns.duplicated()].tolist()
        raise ValueError(f"Duplicate column names found: {dupes[:5]}. Rename them and re-load.")
    # pandas renames repeated headers to "name.1", "name.2"; surface that instead of hiding it.
    mangled = [c for c in df.columns if re.fullmatch(r".+\.\d+", c) and c.rsplit(".", 1)[0] in df.columns]
    if mangled:
        notes.append(f"Duplicate column headers were renamed by the CSV reader: {', '.join(mangled[:5])}. "
                     "Check the source file.")
    if optimize:
        df = optimize_memory(df)
    if sep != ",":
        notes.append(f"Detected {'tab' if sep == chr(9) else repr(sep)} as the column separator.")
    synthetic = is_bundled_synthetic(df, name)
    if synthetic:
        notes.append("SYNTHETIC: this is CreditIQ's bundled demonstration data, not real borrowers.")
    info = DatasetInfo(
        name=sanitize_filename(name, "dataset.csv"),
        source=source_kind,
        is_synthetic=synthetic,
        n_rows_loaded=len(df),
        n_cols=df.shape[1],
        n_rows_file=n_rows_file,
        sampled=sampled,
        notes=notes,
    )
    info.fingerprint = dataframe_fingerprint(df, info.name)
    return df, info


def read_csv_file(
    source: bytes | IO[bytes],
    *,
    name: str | None = None,
    sample_rows: int | None = None,
    seed: int = defaults.DEFAULT_SEED,
    optimize_memory: bool = True,
    source_kind: str = "upload",
) -> tuple[pd.DataFrame, DatasetInfo]:
    """Parse a CSV held in memory (bytes or a binary file-like object), optionally sampling rows.

    For large files on disk, use the trainer's path reader, which streams the file in chunks.
    """
    stream: IO[bytes] = io.BytesIO(source) if isinstance(source, bytes) else source
    notes: list[str] = []
    name = name or "uploaded.csv"
    sep, encoding = sniff(_head_bytes(stream))
    df = _read(stream, sep, encoding)
    n_rows_file = len(df)
    sampled = False
    if sample_rows and len(df) > sample_rows:
        df = df.sample(n=sample_rows, random_state=seed).reset_index(drop=True)
        sampled = True
        notes.append(f"Random sample of {sample_rows:,} of {n_rows_file:,} rows (seed {seed}).")
    return finalize_frame(df, name=name, sep=sep, source_kind=source_kind, n_rows_file=n_rows_file,
                          sampled=sampled, notes=notes, optimize=optimize_memory)


def suggest_target_definition(df: pd.DataFrame, target_col: str, is_synthetic: bool) -> str:
    """Pre-fill text for the target-definition field (user can edit)."""
    if is_synthetic and target_col == "TARGET":
        return SYNTHETIC_TARGET_DEFINITION
    if target_col == "TARGET" and {"SK_ID_CURR", "AMT_CREDIT", "EXT_SOURCE_2"}.issubset(df.columns):
        return HOME_CREDIT_TARGET_DEFINITION
    return ""


def columns_preview(columns: list[str] | pd.Index, limit: int = 12) -> str:
    cols = [str(c) for c in columns]
    head = ", ".join(cols[:limit])
    return head + (f" … (+{len(cols) - limit} more)" if len(cols) > limit else "")


def random_row_mask(n: int, frac: float, rng: np.random.Generator) -> np.ndarray:
    """Row-keep mask used by chunked samplers (same draw order as v1)."""
    return rng.random(n) < frac
