"""Probability calibration fitted on the VALIDATION split only.

A classifier's ``predict_proba`` output is a *score*: class weighting, tree
averaging and regularisation all distort it. Turning it into a probability of
default (PD) needs a calibration map fitted on data the model did not train on.
The final test split is never used to fit the calibrator.
"""
from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

CALIBRATION_METHODS = {
    "sigmoid": "Platt scaling (logistic on the score's log-odds)",
    "isotonic": "Isotonic regression (monotone, non-parametric; needs more data)",
    "none": "No calibration (raw model score is reported as-is)",
}

_EPS = 1e-6


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), _EPS, 1 - _EPS)
    return np.log(p / (1 - p))


class ProbabilityCalibrator:
    """Maps raw model scores to calibrated PDs."""

    def __init__(self, method: str = "sigmoid"):
        if method not in CALIBRATION_METHODS:
            raise ValueError(f"Unknown calibration method '{method}'.")
        self.method = method
        self.fitted_method_ = method
        self.warning_: str | None = None
        self._model = None

    def fit(self, scores, y) -> "ProbabilityCalibrator":
        scores = np.asarray(scores, dtype=float)
        y = np.asarray(y).astype(int)
        if self.method == "none":
            self.fitted_method_ = "none"
            return self
        if len(np.unique(y)) < 2 or len(y) < 20:
            self.fitted_method_ = "none"
            self.warning_ = "Calibration skipped: the validation split has too few rows or only one class."
            return self
        if self.method == "isotonic" and min(np.bincount(y)) < 50:
            self.warning_ = "Few validation defaults for isotonic calibration; Platt scaling was used instead."
            self.fitted_method_ = "sigmoid"
        else:
            self.fitted_method_ = self.method
        if self.fitted_method_ == "sigmoid":
            self._model = LogisticRegression(C=1e6, max_iter=1000)
            self._model.fit(_logit(scores).reshape(-1, 1), y)
        else:
            self._model = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
            self._model.fit(scores, y)
        return self

    def transform(self, scores) -> np.ndarray:
        scores = np.asarray(scores, dtype=float)
        if self.fitted_method_ == "none" or self._model is None:
            return np.clip(scores, 0.0, 1.0)
        if self.fitted_method_ == "sigmoid":
            return self._model.predict_proba(_logit(scores).reshape(-1, 1))[:, 1]
        return np.clip(self._model.predict(scores), 0.0, 1.0)

    def describe(self) -> str:
        text = CALIBRATION_METHODS.get(self.fitted_method_, self.fitted_method_)
        return text + (f" — {self.warning_}" if self.warning_ else "")
