"""Human-readable number formatting (pure functions; the currency is an argument)."""
from __future__ import annotations

import math
from typing import Any


def _bad(v: Any) -> bool:
    return v is None or (isinstance(v, float) and not math.isfinite(v))


def pct(v: float | None, digits: int = 1) -> str:
    return "n/a" if _bad(v) else f"{float(v):.{digits}%}"  # type: ignore[arg-type]


def num(v: float | None, digits: int = 0) -> str:
    return "n/a" if _bad(v) else f"{float(v):,.{digits}f}"  # type: ignore[arg-type]


def dec(v: float | None, digits: int = 3) -> str:
    return "n/a" if _bad(v) else f"{float(v):.{digits}f}"  # type: ignore[arg-type]


def val(v: Any) -> str:
    """Readable value for an input shown next to an explanation."""
    if isinstance(v, str):
        return v
    if _bad(v):
        return "missing"
    v = float(v)
    if math.isnan(v):
        return "missing"
    if abs(v) >= 1000 or v == int(v):
        return f"{v:,.0f}"
    return f"{v:.4g}"


def currency_prefix(currency: str = "") -> str:
    """A one-character symbol is attached ('₹' → '₹1.2M'); codes and abbreviations get a space
    ('INR' → 'INR 1.2M', 'Rs.' → 'Rs. 1.2M')."""
    cur = (currency or "").strip()
    return f"{cur} " if cur and (len(cur) > 1 or cur.isalpha()) else cur


def money(v: float | None, compact: bool = True, currency: str = "") -> str:
    if _bad(v):
        return "n/a"
    cur = currency_prefix(currency)
    x = float(v)  # type: ignore[arg-type]
    sign = "−" if x < 0 else ""
    a = abs(x)
    if compact:
        for div, suf in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")):
            if a >= div:
                return f"{sign}{cur}{a / div:,.2f}{suf}"
    return f"{sign}{cur}{a:,.0f}"


def amount_label(what: str = "Amount", currency: str = "") -> str:
    """Axis title for money values, using the chosen currency (or 'currency units' when none is set)."""
    cur = (currency or "").strip()
    return f"{what} ({cur})" if cur else f"{what} (currency units)"
