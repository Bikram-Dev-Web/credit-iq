"""Cost-sensitive lending-policy comparison.

Decision convention: an applicant is *flagged* when the policy does not
auto-approve them (PD ≥ threshold, or negative expected contribution). Flagged
applicants incur a review cost and, in this simplified cost matrix, are treated
as not booked after review.

Cost matrix (all user-declared assumptions):

==================  ======================  ==========================
                    Actually defaulted      Actually repaid
==================  ======================  ==========================
Flagged             TP: review cost         FP: review + lost margin
Approved            FN: missed-default cost TN: 0
==================  ======================  ==========================

A missed default is not necessarily a loss of the full loan amount (recoveries
reduce it — hence LGD × EAD in the per-loan mode), and rejecting a good
applicant does not create a booked accounting loss: it is an opportunity cost
whose size depends on the lender's margin and on whether the customer would
have taken the loan at all.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from .financial_calculations import horizon_pd


@dataclass
class CostAssumptions:
    mode: str = "per_loan"  # "per_loan" (scaled by each loan's EAD) or "fixed"
    missed_default_cost: float = 200_000.0  # fixed mode: cost of approving a defaulter
    rejected_good_cost: float = 40_000.0  # fixed mode: cost of flagging a good applicant
    review_cost: float = 500.0  # per flagged application, both modes
    lgd: float = 0.45  # per-loan mode: missed default costs LGD × EAD
    lost_margin_rate: float = 0.08  # per-loan mode: rejected good applicant costs margin × EAD

    def validate(self) -> None:
        if self.mode not in ("per_loan", "fixed"):
            raise ValueError("Cost mode must be 'per_loan' or 'fixed'.")
        for name in ("missed_default_cost", "rejected_good_cost", "review_cost", "lost_margin_rate"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name.replace('_', ' ')} cannot be negative.")
        if not 0 <= self.lgd <= 1:
            raise ValueError("LGD must be between 0 and 1.")

    def to_dict(self) -> dict:
        return asdict(self)


def cost_vectors(costs: CostAssumptions, ead: np.ndarray | None, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-applicant cost of a missed default (FN) and of a rejected good applicant (FP)."""
    costs.validate()
    if costs.mode == "per_loan":
        if ead is None:
            raise ValueError("Per-loan costs need an exposure (EAD) for each applicant.")
        e = np.asarray(ead, dtype=float)
        if len(e) != n:
            raise ValueError("EAD length does not match the number of applicants.")
        return costs.lgd * e, costs.lost_margin_rate * e
    return np.full(n, costs.missed_default_cost, dtype=float), np.full(n, costs.rejected_good_cost, dtype=float)


def evaluate_policy(y, flagged, c_fn, c_fp, review_cost: float) -> dict:
    y = np.asarray(y).astype(int)
    f = np.asarray(flagged).astype(bool)
    tp = int((f & (y == 1)).sum())
    fp = int((f & (y == 0)).sum())
    tn = int((~f & (y == 0)).sum())
    fn = int((~f & (y == 1)).sum())
    missed = float(np.sum(np.asarray(c_fn)[~f & (y == 1)]))
    rejected = float(np.sum(np.asarray(c_fp)[f & (y == 0)]))
    review = float(review_cost * f.sum())
    total = missed + rejected + review
    n = len(y)
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn, "flagged": int(f.sum()),
            "approval_rate": (n - int(f.sum())) / n if n else None,
            "default_rate_among_approved": fn / (fn + tn) if (fn + tn) else None,
            "missed_default_cost": missed, "rejected_good_cost": rejected, "review_cost_total": review,
            "total_cost": total, "cost_per_applicant": total / n if n else None}


def cost_curve(y, pd_, c_fn, c_fp, review_cost: float, thresholds=None) -> pd.DataFrame:
    if thresholds is None:
        thresholds = np.round(np.linspace(0.0, 1.0, 101), 3)
    p = np.asarray(pd_, dtype=float)
    rows = []
    for t in thresholds:
        r = evaluate_policy(y, p >= t, c_fn, c_fp, review_cost)
        rows.append({"threshold": float(t), **r})
    return pd.DataFrame(rows)


def cost_optimal_threshold(y_val, pd_val, c_fn, c_fp, review_cost: float) -> float:
    """Threshold minimising total declared cost on the VALIDATION split (ties → lowest threshold)."""
    p = np.asarray(pd_val, dtype=float)
    grid = np.unique(np.concatenate([np.round(np.linspace(0, 1, 201), 4), np.quantile(p, np.linspace(0, 1, 201))]))
    curve = cost_curve(y_val, p, c_fn, c_fp, review_cost, grid)
    return float(curve.loc[curve["total_cost"].idxmin(), "threshold"])


def expected_contribution_per_loan(pd_, ead, lgd: float, annual_rate: float, funding_rate: float,
                                   opex_per_loan: float, horizon_years: float = 1.0,
                                   interest_recovery_on_default: float = 0.0) -> np.ndarray:
    """E[contribution] per applicant — same conventions as ``financial_calculations.lending_contribution``
    (annual PD converted to the horizon, annual operating cost × horizon)."""
    p = np.asarray(pd_, dtype=float)
    e = np.asarray(ead, dtype=float)
    q = horizon_pd(p, horizon_years)
    h = horizon_years
    interest = e * annual_rate * h * ((1 - q) + q * interest_recovery_on_default)
    return interest - e * funding_rate * h - q * lgd * e - opex_per_loan * h


def realized_contribution(y, approved, ead, lgd: float, annual_rate: float, funding_rate: float,
                          opex_per_loan: float, horizon_years: float = 1.0,
                          interest_recovery_on_default: float = 0.0) -> float:
    """Back-test: contribution of the approved book using OBSERVED outcomes and the ASSUMED LGD/rates.
    An observed default is treated as a default within the horizon."""
    y = np.asarray(y).astype(int)
    a = np.asarray(approved).astype(bool)
    e = np.asarray(ead, dtype=float)
    good = a & (y == 0)
    bad = a & (y == 1)
    h = horizon_years
    income_good = np.sum(e[good] * (annual_rate - funding_rate) * h)
    income_bad = np.sum(e[bad] * (annual_rate * h * interest_recovery_on_default - funding_rate * h - lgd))
    return float(income_good + income_bad - opex_per_loan * h * a.sum())


def compare_policies(
    y_val, pd_val, ead_val, y_test, pd_test, ead_test, fixed_threshold: float, costs: CostAssumptions,
    annual_rate: float, funding_rate: float, opex_per_loan: float, horizon_years: float = 1.0,
) -> tuple[pd.DataFrame, dict]:
    """Evaluate four policies on the TEST split. Anything tuned is tuned on VALIDATION only."""
    y_test = np.asarray(y_test).astype(int)
    n = len(y_test)
    c_fn_v, c_fp_v = cost_vectors(costs, ead_val, len(np.asarray(y_val)))
    c_fn_t, c_fp_t = cost_vectors(costs, ead_test, n)
    opt_t = cost_optimal_threshold(y_val, pd_val, c_fn_v, c_fp_v, costs.review_cost)
    exp_contrib = expected_contribution_per_loan(pd_test, ead_test, costs.lgd, annual_rate, funding_rate,
                                                 opex_per_loan, horizon_years)
    policies = {
        "Baseline — approve all (no model)": np.zeros(n, dtype=bool),
        f"Fixed PD threshold ({fixed_threshold:.1%})": np.asarray(pd_test) >= fixed_threshold,
        f"Cost-minimising threshold ({opt_t:.1%}, chosen on validation)": np.asarray(pd_test) >= opt_t,
        "Expected-contribution rule (approve if E[contribution] > 0)": exp_contrib <= 0,
    }
    rows = []
    for name, flagged in policies.items():
        r = evaluate_policy(y_test, flagged, c_fn_t, c_fp_t, costs.review_cost)
        r["realized_contribution_approved_book"] = realized_contribution(
            y_test, ~flagged, ead_test, costs.lgd, annual_rate, funding_rate, opex_per_loan, horizon_years)
        rows.append({"policy": name, **r})
    table = pd.DataFrame(rows)
    return table, {"cost_optimal_threshold": opt_t}
