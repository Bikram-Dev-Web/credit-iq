"""Provenance vocabulary: where a number comes from.

Every number CreditIQ shows or returns carries one or more of these labels, so an observed
default rate, a model output, a user assumption and a simulation are never confused.
"""
from __future__ import annotations

from enum import StrEnum


class Provenance(StrEnum):
    OBSERVED = "observed"
    MEASURED = "measured"
    MODEL = "model"
    ASSUMPTION = "assumption"
    SIMULATED = "simulated"
    SYNTHETIC = "synthetic"


PROVENANCE_LABELS: dict[Provenance, tuple[str, str]] = {
    Provenance.OBSERVED: ("Observed", "Historical fact from the loaded data"),
    Provenance.MEASURED: ("Measured on held-out data", "Computed by evaluating the model on data it did not train on"),
    Provenance.MODEL: ("Model output", "Produced by the trained model"),
    Provenance.ASSUMPTION: ("Assumption", "Value entered by the user"),
    Provenance.SIMULATED: ("Simulated", "Projection combining model outputs with assumptions"),
    Provenance.SYNTHETIC: ("Synthetic data", "Computed on fabricated demonstration data"),
}
