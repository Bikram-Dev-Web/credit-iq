"""Financial management engine: expected loss, lending contribution, break-even, stress and NPV.

Conventions (shown in the UI as well):

* **PD is treated as an annual (12-month) probability.** A model trained on a
  different event or window (e.g. Home Credit's early-instalment payment
  difficulty) only approximates a 12-month PD — that mapping is an *assumption*.
* Over a horizon of *h* years the default probability is
  ``PD_h = 1 − (1 − PD)^h`` (constant annual hazard). Every horizon-dependent
  quantity — interest, funding cost, operating cost and expected loss — uses the
  same *h* and the same ``PD_h``, so horizons cannot drift apart.
* LGD is the fraction of EAD lost after recoveries; in the one-period view,
  recoveries arrive within the horizon and are not discounted (the NPV module
  discounts them).
* EAD is the exposure outstanding at default (loan amount × CCF).
* ``EL = PD × LGD × EAD`` is a simplified credit-loss estimate. It is not a full
  IFRS 9 ECL (no staging, lifetime PD term structure or forward-looking macro
  weighting) and not a regulatory capital calculation.
* Operating cost is an annual cost per loan.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from typing import Any, overload

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

ArrayIn = np.ndarray | pd.Series | Sequence[float]


def _check_unit_interval(name: str, value: ArrayLike) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if np.any(~np.isfinite(arr)):
        raise ValueError(f"{name} must be a finite number.")
    if np.any((arr < 0) | (arr > 1)):
        raise ValueError(f"{name} must be between 0 and 1 (got {arr.min():.4g}–{arr.max():.4g}).")
    return arr


def _check_non_negative(name: str, value: ArrayLike) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if np.any(~np.isfinite(arr)):
        raise ValueError(f"{name} must be a finite number.")
    if np.any(arr < 0):
        raise ValueError(f"{name} cannot be negative.")
    return arr


@overload
def horizon_pd(annual_pd: float, horizon_years: float = 1.0) -> float: ...
@overload
def horizon_pd(annual_pd: ArrayIn, horizon_years: float = 1.0) -> np.ndarray: ...
def horizon_pd(annual_pd: ArrayLike, horizon_years: float = 1.0) -> Any:
    """Probability of default within ``horizon_years`` from an annual PD: 1 − (1 − PD)^h."""
    p = _check_unit_interval("PD", annual_pd)
    if not (0 < horizon_years <= 30):
        raise ValueError("Horizon must be between 0 and 30 years.")
    out = 1.0 - (1.0 - p) ** horizon_years
    return float(out) if out.ndim == 0 else out


@overload
def annual_pd_from_horizon(pd_h: float, horizon_years: float = 1.0) -> float: ...
@overload
def annual_pd_from_horizon(pd_h: ArrayIn, horizon_years: float = 1.0) -> np.ndarray: ...
def annual_pd_from_horizon(pd_h: ArrayLike, horizon_years: float = 1.0) -> Any:
    """Inverse of :func:`horizon_pd`."""
    p = _check_unit_interval("Horizon PD", pd_h)
    out = 1.0 - (1.0 - p) ** (1.0 / horizon_years)
    return float(out) if out.ndim == 0 else out


@overload
def expected_loss(pd_: float, lgd: float, ead: float) -> float: ...
@overload
def expected_loss(pd_: ArrayIn, lgd: ArrayLike, ead: ArrayLike) -> np.ndarray: ...
@overload
def expected_loss(pd_: ArrayLike, lgd: ArrayLike, ead: ArrayLike) -> Any: ...
def expected_loss(pd_: ArrayLike, lgd: ArrayLike, ead: ArrayLike) -> Any:
    """EL = PD × LGD × EAD (element-wise). Returns a float for scalar input."""
    p = _check_unit_interval("PD", pd_)
    l = _check_unit_interval("LGD", lgd)
    e = _check_non_negative("EAD", ead)
    el = p * l * e
    return float(el) if el.ndim == 0 else el


def portfolio_expected_loss(pd_: ArrayLike, lgd: ArrayLike, ead: ArrayLike) -> dict[str, Any]:
    """Sum of loan-level expected losses, each using its own PD, LGD and EAD."""
    el = np.atleast_1d(expected_loss(pd_, lgd, ead))
    e = np.broadcast_to(np.asarray(ead, dtype=float), el.shape)
    total_ead = float(e.sum())
    total_el = float(el.sum())
    return {"n_loans": int(el.size), "total_ead": total_ead, "total_el": total_el,
            "el_rate": total_el / total_ead if total_ead else None,
            "ead_weighted_pd": float(np.sum(np.broadcast_to(np.asarray(pd_, dtype=float), el.shape) * e) / total_ead) if total_ead else None}


@dataclass
class LendingAssumptions:
    n_loans: int = 1000
    avg_exposure: float = 500_000.0
    pd: float = 0.08  # annual (12-month) PD
    lgd: float = 0.45
    annual_rate: float = 0.16
    funding_rate: float = 0.07
    opex_per_loan: float = 1_500.0  # per loan per year
    horizon_years: float = 1.0
    upfront_fee_rate: float = 0.0
    interest_recovery_on_default: float = 0.0  # share of contractual interest still collected from defaulters

    def validate(self) -> None:
        if self.n_loans < 0 or int(self.n_loans) != self.n_loans:
            raise ValueError("Number of loans must be a non-negative integer.")
        _check_non_negative("Average exposure", self.avg_exposure)
        _check_unit_interval("PD", self.pd)
        _check_unit_interval("LGD", self.lgd)
        _check_non_negative("Operating cost", self.opex_per_loan)
        _check_unit_interval("Interest collected from defaulters", self.interest_recovery_on_default)
        for name in ("annual_rate", "funding_rate", "upfront_fee_rate"):
            v = getattr(self, name)
            if not np.isfinite(v) or v < 0 or v > 2:
                raise ValueError(f"{name.replace('_', ' ').title()} must be between 0% and 200%.")
        if not (0 < self.horizon_years <= 30):
            raise ValueError("Horizon must be between 0 and 30 years.")

    @property
    def pd_horizon(self) -> float:
        return horizon_pd(self.pd, self.horizon_years)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def lending_contribution(a: LendingAssumptions) -> dict[str, Any]:
    """Simplified expected contribution over the horizon *h*.

    Expected Contribution = Expected Interest Income + Fees − Funding Cost
                            − Expected Credit Loss − Operating Cost, with
    ``PD_h = 1 − (1 − PD)^h`` and

    * interest  = rate × EAD × h × [(1 − PD_h) + PD_h × k]   (k = share of interest defaulters still pay)
    * funding   = funding rate × EAD × h                     (full exposure funded for the whole horizon)
    * ECL       = PD_h × LGD × EAD                           (principal loss — counted once)
    * opex      = annual operating cost per loan × h

    Principal repayment is a return of capital, not income. No taxes, capital
    charges, prepayment, amortisation or discounting.
    """
    a.validate()
    exposure = a.n_loans * a.avg_exposure
    h = a.horizon_years
    p_h = a.pd_horizon
    interest = exposure * a.annual_rate * h * ((1 - p_h) + p_h * a.interest_recovery_on_default)
    fees = exposure * a.upfront_fee_rate
    funding = exposure * a.funding_rate * h
    ecl = a.n_loans * expected_loss(p_h, a.lgd, a.avg_exposure)
    opex = a.n_loans * a.opex_per_loan * h
    contribution = interest + fees - funding - ecl - opex
    return {
        "total_exposure": exposure,
        "pd_over_horizon": p_h,
        "expected_interest_income": interest,
        "fee_income": fees,
        "funding_cost": funding,
        "expected_credit_loss": ecl,
        "operating_cost": opex,
        "expected_contribution": contribution,
        "contribution_per_loan": contribution / a.n_loans if a.n_loans else 0.0,
        "contribution_margin_on_exposure": contribution / exposure if exposure else None,
        "el_rate": ecl / exposure if exposure else None,
        "break_even_pd": break_even_pd(a),
        "break_even_rate": break_even_rate(a),
    }


def break_even_pd(a: LendingAssumptions) -> float | None:
    """Annual PD at which expected contribution is exactly zero.

    Solves r·h·[(1−q) + q·k] + f − c·h − q·LGD − opex·h/EAD = 0 for the horizon
    PD q, then converts q back to an annual PD. Returns None when no PD in
    [0, 1] breaks even (always profitable, or never).
    """
    if a.avg_exposure <= 0:
        return None
    h, k = a.horizon_years, a.interest_recovery_on_default
    numer = a.annual_rate * h + a.upfront_fee_rate - a.funding_rate * h - a.opex_per_loan * h / a.avg_exposure
    denom = a.annual_rate * h * (1 - k) + a.lgd
    if denom <= 0:
        return None
    q = numer / denom
    return float(annual_pd_from_horizon(q, h)) if 0 <= q <= 1 else None


def break_even_rate(a: LendingAssumptions) -> float | None:
    """Annual lending rate that makes expected contribution zero at the assumed PD (risk-based pricing floor)."""
    if a.avg_exposure <= 0:
        return None
    h, k = a.horizon_years, a.interest_recovery_on_default
    q = a.pd_horizon
    paying = h * ((1 - q) + q * k)
    if paying <= 0:
        return None
    return float((a.funding_rate * h + q * a.lgd + a.opex_per_loan * h / a.avg_exposure - a.upfront_fee_rate) / paying)


def sensitivity_analysis(base: LendingAssumptions, shock: float = 0.20) -> pd.DataFrame:
    """One-at-a-time ±shock (relative) on each driver → change in expected contribution (tornado data)."""
    base_c = lending_contribution(base)["expected_contribution"]
    drivers = {"pd": "PD", "lgd": "LGD", "avg_exposure": "Average exposure", "annual_rate": "Lending rate",
               "funding_rate": "Funding rate", "opex_per_loan": "Operating cost per loan"}
    rows = []
    for attr, label in drivers.items():
        val = getattr(base, attr)
        out = {}
        for sign, name in ((-1, "low"), (1, "high")):
            new_val = val * (1 + sign * shock)
            if attr in ("pd", "lgd"):
                new_val = min(max(new_val, 0.0), 1.0)
            out[name] = lending_contribution(replace(base, **{attr: new_val}))["expected_contribution"] - base_c
        rows.append({"driver": label, "change_if_driver_down": out["low"], "change_if_driver_up": out["high"],
                     "range": abs(out["high"] - out["low"])})
    return pd.DataFrame(rows).sort_values("range", ascending=True).reset_index(drop=True)


DEFAULT_SCENARIOS: list[dict[str, Any]] = [
    {"scenario": "Improved", "pd_multiplier": 0.8, "lgd_shift": -0.05, "ead_multiplier": 1.0},
    {"scenario": "Base", "pd_multiplier": 1.0, "lgd_shift": 0.0, "ead_multiplier": 1.0},
    {"scenario": "Adverse", "pd_multiplier": 1.5, "lgd_shift": 0.10, "ead_multiplier": 1.05},
    {"scenario": "Severe stress", "pd_multiplier": 2.2, "lgd_shift": 0.20, "ead_multiplier": 1.10},
]


def _apply_scenario(pd_: ArrayLike, lgd: ArrayLike, ead: ArrayLike,
                    sc: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    p = np.clip(np.asarray(pd_, dtype=float) * float(sc["pd_multiplier"]), 0, 1)
    l = np.clip(np.asarray(lgd, dtype=float) + float(sc["lgd_shift"]), 0, 1)
    e = np.asarray(ead, dtype=float) * float(sc["ead_multiplier"])
    return p, l, e


def stress_test(pd_: ArrayLike, lgd: ArrayLike, ead: ArrayLike, scenarios: list[dict[str, Any]], annual_rate: float, funding_rate: float,
                opex_per_loan: float, horizon_years: float = 1.0, interest_recovery_on_default: float = 0.0) -> pd.DataFrame:
    """Apply annual-PD multipliers, LGD shifts and EAD multipliers loan by loan (PD and LGD capped at 100%),
    then compute horizon EL and contribution with the same conventions as :func:`lending_contribution`."""
    if not (0 < horizon_years <= 30):
        raise ValueError("Horizon must be between 0 and 30 years.")
    for sc in scenarios:
        if sc["pd_multiplier"] < 0 or sc["ead_multiplier"] < 0:
            raise ValueError("Scenario multipliers cannot be negative.")
    rows = []
    h = horizon_years
    for sc in scenarios:
        p, l, e = _apply_scenario(pd_, lgd, ead, sc)
        p = np.atleast_1d(p)
        l = np.broadcast_to(np.atleast_1d(l), p.shape)
        e = np.broadcast_to(np.atleast_1d(e), p.shape)
        q = horizon_pd(p, h)
        el = q * l * e
        exposure = e.sum()
        interest = float(np.sum(e * annual_rate * h * ((1 - q) + q * interest_recovery_on_default)))
        funding = float(exposure * funding_rate * h)
        opex = float(len(p) * opex_per_loan * h)
        contribution = interest - funding - el.sum() - opex
        rows.append({
            "scenario": sc["scenario"], "pd_multiplier": sc["pd_multiplier"], "lgd_shift": sc["lgd_shift"],
            "ead_multiplier": sc["ead_multiplier"], "n_loans": len(p), "total_ead": float(exposure),
            "avg_pd": float(p.mean()), "avg_pd_over_horizon": float(q.mean()),
            "ead_weighted_pd": float((p * e).sum() / exposure) if exposure else None,
            "avg_lgd": float(l.mean()), "expected_loss": float(el.sum()),
            "el_rate": float(el.sum() / exposure) if exposure else None,
            "expected_interest_income": interest, "funding_cost": funding, "operating_cost": opex,
            "expected_contribution": contribution,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Loan cash flows and NPV
# ---------------------------------------------------------------------------

def annuity_payment(principal: float, annual_rate: float, term_months: int) -> float:
    if term_months <= 0:
        raise ValueError("Term must be at least one month.")
    r = annual_rate / 12
    if r == 0:
        return principal / term_months
    return principal * r / (1 - (1 + r) ** (-term_months))


def amortization_schedule(principal: float, annual_rate: float, term_months: int) -> pd.DataFrame:
    _check_non_negative("Principal", principal)
    pay = annuity_payment(principal, annual_rate, term_months)
    r = annual_rate / 12
    bal = principal
    rows = []
    for m in range(1, term_months + 1):
        interest = bal * r
        princ = pay - interest
        opening = bal
        bal = max(bal - princ, 0.0)
        rows.append({"month": m, "opening_balance": opening, "payment": pay, "interest": interest,
                     "principal": princ, "closing_balance": bal})
    return pd.DataFrame(rows)


def monthly_hazard(annual_pd: float) -> float:
    _check_unit_interval("Annual PD", annual_pd)
    if annual_pd >= 1:
        return 1.0
    return 1 - (1 - annual_pd) ** (1 / 12)


def loan_npv(principal: float, annual_rate: float, term_months: int, annual_pd: float, lgd: float,
             discount_rate: float, recovery_lag_months: int = 6, upfront_fee: float = 0.0,
             servicing_cost_monthly: float = 0.0) -> dict[str, Any]:
    """Risk-adjusted NPV of one amortising loan under a constant default hazard.

    Assumptions: monthly annuity repayments; a constant monthly default hazard
    implied by ``annual_pd``; default happens instead of that month's payment;
    exposure at default = opening balance plus that month's accrued interest;
    the recovery (1 − LGD) × EAD arrives ``recovery_lag_months`` later; no
    prepayment; cash flows discounted monthly at ``discount_rate`` (e.g. the
    lender's funding cost or required return). With LGD = 0, no lag and the
    discount rate equal to the loan rate, default costs nothing in PV terms —
    a useful sanity check.
    """
    _check_unit_interval("LGD", lgd)
    if discount_rate < 0 or discount_rate > 2:
        raise ValueError("Discount rate must be between 0% and 200%.")
    if recovery_lag_months < 0:
        raise ValueError("Recovery lag cannot be negative.")
    sched = amortization_schedule(principal, annual_rate, term_months)
    h = monthly_hazard(annual_pd)
    T = term_months + recovery_lag_months
    months = np.arange(0, T + 1)
    contractual = np.zeros(T + 1)
    payments = np.zeros(T + 1)
    recoveries = np.zeros(T + 1)
    servicing = np.zeros(T + 1)
    contractual[0] = payments[0] = -principal + upfront_fee
    surv_prev = 1.0
    surv = np.ones(term_months + 1)
    for m, opening, interest, payment in zip(sched["month"].to_numpy(dtype=int),
                                             sched["opening_balance"].to_numpy(dtype=float),
                                             sched["interest"].to_numpy(dtype=float),
                                             sched["payment"].to_numpy(dtype=float)):
        default_prob = surv_prev * h
        s = surv_prev * (1 - h)
        surv[m] = s
        payments[m] += s * payment
        contractual[m] += payment
        ead = opening + interest  # balance outstanding plus the missed month's interest
        recoveries[m + recovery_lag_months] += default_prob * (1 - lgd) * ead
        servicing[m] -= surv_prev * servicing_cost_monthly
        contractual[m] -= servicing_cost_monthly
        surv_prev = s
    expected = payments + recoveries + servicing
    disc = (1 + discount_rate / 12) ** (-months)
    npv_expected = float(np.sum(expected * disc))
    npv_contractual = float(np.sum(contractual * disc))

    cf_table = pd.DataFrame({"month": months, "contractual_cash_flow": contractual,
                             "expected_payments": payments, "expected_recoveries": recoveries,
                             "expected_servicing_cost": servicing, "expected_cash_flow": expected,
                             "discount_factor": disc, "pv_expected": expected * disc})
    return {
        "monthly_payment": float(sched["payment"].iloc[0]) if len(sched) else 0.0,
        "monthly_hazard": h,
        "lifetime_pd": float(1 - surv[-1]),
        "npv_contractual": npv_contractual,
        "npv_risk_adjusted": npv_expected,
        "pv_credit_cost": npv_contractual - npv_expected,
        "expected_annual_yield": _irr_annual(expected),
        "contractual_annual_yield": _irr_annual(contractual),
        "cash_flows": cf_table,
    }


def _irr_annual(cash_flows: np.ndarray) -> float | None:
    """Annualised (monthly-compounded) IRR by bisection; None if no sign change."""
    cf = np.asarray(cash_flows, dtype=float)
    t = np.arange(len(cf))

    def npv(r_m: float) -> float:
        return float(np.sum(cf / (1 + r_m) ** t))

    lo, hi = -0.99, 1.0
    f_lo, f_hi = npv(lo), npv(hi)
    if np.sign(f_lo) == np.sign(f_hi):
        return None
    for _ in range(200):
        mid = (lo + hi) / 2
        f_mid = npv(mid)
        if abs(f_mid) < 1e-9:
            break
        if np.sign(f_mid) == np.sign(f_lo):
            lo, f_lo = mid, f_mid
        else:
            hi = mid
    return float((1 + (lo + hi) / 2) ** 12 - 1)
