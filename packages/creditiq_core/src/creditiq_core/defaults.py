"""Domain defaults shared by every CreditIQ service.

Plain constants only: no environment variables, paths or other side effects. Services
read their own settings (paths, database URLs, secrets) from the environment.
"""
from __future__ import annotations

DEFAULT_SEED = 42

# Training is laptop-friendly by default; the trainer config can raise it.
DEFAULT_TRAIN_SAMPLE = 60_000

# Financial assumptions (illustrative, always editable).
DEFAULT_LGD = 0.45
DEFAULT_CCF = 1.00
DEFAULT_ANNUAL_RATE = 0.16
DEFAULT_FUNDING_RATE = 0.07
DEFAULT_OPEX_PER_LOAN = 1_500.0

# Risk bands: PD cut points and labels (len(labels) == len(cutoffs) + 1).
DEFAULT_BAND_CUTOFFS: tuple[float, ...] = (0.05, 0.10, 0.20)
BAND_LABELS: tuple[str, ...] = ("Low", "Moderate", "Elevated", "High")

# Population Stability Index rule-of-thumb thresholds used in monitoring.
PSI_MODERATE = 0.10
PSI_SIGNIFICANT = 0.25

APP_DISCLAIMER = (
    "CreditIQ is an educational decision-support prototype. It does not make, "
    "and must not be used to make, real credit decisions."
)
