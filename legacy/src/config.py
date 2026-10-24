"""Central configuration: paths, defaults and constants.

Paths can be overridden with environment variables (see ``.env.example``).
A minimal ``.env`` loader is included so no extra dependency is needed.
"""
from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv(path: Path) -> None:
    """Load KEY=VALUE pairs from a local .env file without overriding the shell."""
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_dotenv(PROJECT_ROOT / ".env")


def _path_from_env(var: str, default: Path) -> Path:
    value = os.environ.get(var)
    return Path(value).expanduser().resolve() if value else default


# v2 repo: the synthetic sample lives in the shared top-level data/ folder.
DATA_DIR = _path_from_env("CREDITIQ_DATA_DIR", PROJECT_ROOT.parent / "data")
MODELS_DIR = _path_from_env("CREDITIQ_MODELS_DIR", PROJECT_ROOT / "models")
REPORTS_DIR = _path_from_env("CREDITIQ_REPORTS_DIR", PROJECT_ROOT / "reports")
STORAGE_DIR = _path_from_env("CREDITIQ_STORAGE_DIR", PROJECT_ROOT / "storage")
DOCS_DIR = PROJECT_ROOT / "docs"
DB_PATH = STORAGE_DIR / "creditiq.db"

SAMPLE_DATA_FILE = DATA_DIR / "sample_credit_data.csv"
SAMPLE_MONITORING_FILE = DATA_DIR / "sample_monitoring_batch.csv"

# Upload guard rails (MB). Streamlit's own limit is set in .streamlit/config.toml.
MAX_UPLOAD_MB = int(os.environ.get("CREDITIQ_MAX_UPLOAD_MB", "400"))

DEFAULT_SEED = 42

# Default training sample keeps a laptop responsive on the 307k-row Home Credit file.
DEFAULT_TRAIN_SAMPLE = 60_000

# Illustrative financial defaults. Every one of these is a user-editable assumption.
DEFAULT_LGD = 0.45
DEFAULT_CCF = 1.00
DEFAULT_ANNUAL_RATE = 0.16
DEFAULT_FUNDING_RATE = 0.07
DEFAULT_OPEX_PER_LOAN = 1_500.0

# Risk-band cutoffs on the calibrated PD (lower bounds of Moderate, Elevated, High).
DEFAULT_BAND_CUTOFFS = (0.05, 0.10, 0.20)
BAND_LABELS = ("Low", "Moderate", "Elevated", "High")

# Population Stability Index rule-of-thumb thresholds used in monitoring.
PSI_MODERATE = 0.10
PSI_SIGNIFICANT = 0.25

APP_DISCLAIMER = (
    "CreditIQ is an academic and analytical demonstration. It does not make, "
    "and must not be used to make, real credit decisions."
)


def ensure_dirs() -> None:
    """Create writable runtime folders if they do not exist."""
    for folder in (DATA_DIR, MODELS_DIR, REPORTS_DIR, STORAGE_DIR):
        folder.mkdir(parents=True, exist_ok=True)
