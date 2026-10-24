"""What a trained model version ships with, and the held-out evaluation snapshot kept beside it.

``ModelArtifact`` is everything the scoring service needs: the fitted pipeline and calibrator,
the input schema, the validation-chosen threshold, the monitoring reference distribution and
the training-mean vector used for exact linear explanations. It contains **no data rows** —
neither training rows nor held-out labels.

``EvaluationSnapshot`` holds the scored validation and test rows that portfolio, fairness and
policy-simulation analytics need. It is stored as a separate artifact, never inside the model,
and it never contains training rows.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from creditiq_core.models.credit_model import CreditModel

ARTIFACT_FORMAT_VERSION = 2
SNAPSHOT_SPLITS = ("validation", "test")


@dataclass
class ModelArtifact:
    model: CreditModel
    input_schema: dict[str, dict[str, Any]]
    threshold: float
    threshold_rule: str
    reference: dict[str, Any]
    metadata: dict[str, Any]
    transformed_mean: np.ndarray | None = None
    threshold_source: str = "validation"
    format_version: int = ARTIFACT_FORMAT_VERSION

    @property
    def input_columns(self) -> list[str]:
        return list(self.model.input_columns)

    @property
    def is_synthetic(self) -> bool:
        return bool(self.metadata.get("dataset", {}).get("is_synthetic", False))

    def predict_pd(self, X: pd.DataFrame | pd.Series | dict[str, Any]) -> np.ndarray:
        return self.model.predict_pd(X)

    def predict_score(self, X: pd.DataFrame | pd.Series | dict[str, Any]) -> np.ndarray:
        return self.model.predict_score(X)


@dataclass
class EvaluationSnapshot:
    """Scored held-out rows (validation + test) for one model version.

    ``frame`` has one row per held-out applicant with the columns ``split``, ``y`` (observed
    label, 1 = default under the confirmed mapping), ``pd``, ``raw_score`` and the retained
    source columns (model inputs, exposure, date and the sensitive attributes kept for the
    fairness audit). Identifiers and excluded columns are dropped.
    """

    frame: pd.DataFrame
    model_key: str
    threshold: float
    exposure_col: str | None
    date_col: str | None
    sensitive_cols: list[str] = field(default_factory=list)
    predictors: list[str] = field(default_factory=list)

    def split(self, name: str) -> pd.DataFrame:
        if name not in SNAPSHOT_SPLITS:
            raise ValueError(f"The evaluation snapshot holds only {SNAPSHOT_SPLITS}; '{name}' is not available.")
        return self.frame[self.frame["split"] == name].reset_index(drop=True)

    def y(self, name: str) -> np.ndarray:
        return self.split(name)["y"].to_numpy(dtype=int)

    def pd(self, name: str) -> np.ndarray:
        return self.split(name)["pd"].to_numpy(dtype=float)
