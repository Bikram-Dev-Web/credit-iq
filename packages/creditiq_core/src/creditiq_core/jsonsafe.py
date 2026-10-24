"""JSON serialisation that always produces valid JSON (numpy-aware; NaN and ±inf become null)."""
from __future__ import annotations

import json
from datetime import date, datetime
from typing import Any

import numpy as np
import pandas as pd


def _default(o: Any) -> Any:
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return sanitize(o.tolist())
    if isinstance(o, pd.Timestamp | datetime | date):
        return str(o)
    return str(o)


def sanitize(obj: Any) -> Any:
    """Recursively replace non-finite floats with ``None`` and numpy containers with lists."""
    if isinstance(obj, dict):
        return {str(k): sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [sanitize(v) for v in obj]
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    if isinstance(obj, np.floating):
        return None if not np.isfinite(obj) else float(obj)
    if isinstance(obj, np.ndarray):
        return sanitize(obj.tolist())
    return obj


def to_json(obj: Any, indent: int | None = None, sort_keys: bool = False) -> str:
    return json.dumps(sanitize(obj), default=_default, indent=indent, sort_keys=sort_keys, ensure_ascii=False)
