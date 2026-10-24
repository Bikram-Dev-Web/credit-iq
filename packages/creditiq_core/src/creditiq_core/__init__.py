"""CreditIQ core library.

Pure Python: data validation and target mapping, features, PD models, calibration,
evaluation, finance (expected loss, economics, cost matrix, portfolio), fairness,
monitoring and explainability. No Streamlit, no web framework, and no I/O side effects:
services load and save; the core computes.
"""
from creditiq_core.provenance import Provenance

__version__ = "2.0.0.dev0"

__all__ = ["Provenance", "__version__"]
