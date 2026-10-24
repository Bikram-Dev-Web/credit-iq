"""Simulate a policy on held-out applicants and compare it with another policy.

Run on the VALIDATION split by default: that is where choices are made. The test split is a
read-only final check, and iterating on test results would turn the test set into a tuning set.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from creditiq_core.finance.cost_matrix import (
    CostAssumptions,
    cost_vectors,
    evaluate_policy,
    expected_contribution_per_loan,
    realized_contribution,
)
from creditiq_core.finance.economics import expected_loss, horizon_pd
from creditiq_core.policy.engine import apply_policy
from creditiq_core.policy.schema import Policy


def exposure(frame: pd.DataFrame, exposure_col: str | None, policy: Policy) -> tuple[np.ndarray, int]:
    """EAD per row (exposure × CCF; missing exposure → policy default). Returns (ead, n_imputed)."""
    if exposure_col and exposure_col in frame.columns:
        e = pd.to_numeric(frame[exposure_col], errors="coerce")
        n_missing = int(e.isna().sum())
        values = e.fillna(policy.economics.default_exposure).clip(lower=0).to_numpy(dtype=float)
    else:
        n_missing = len(frame)
        values = np.full(len(frame), policy.economics.default_exposure)
    return values * policy.economics.ccf, n_missing


def simulate(policy: Policy, frame: pd.DataFrame, model_threshold: float, exposure_col: str | None,
             pd_col: str = "pd", y_col: str = "y") -> dict[str, Any]:
    """Outcome, loss and contribution metrics of ``policy`` on held-out rows with observed labels."""
    p = frame[pd_col].to_numpy(dtype=float)
    y = frame[y_col].to_numpy(dtype=int)
    res = apply_policy(policy, frame, p, model_threshold)
    flagged = res["flagged"].to_numpy(dtype=bool)
    within = ~flagged
    ead, n_imputed = exposure(frame, exposure_col, policy)
    eco = policy.economics
    el_12m = np.asarray(expected_loss(p, eco.lgd, ead), dtype=float)
    q = np.asarray(horizon_pd(p, eco.horizon_years), dtype=float)
    el_h = q * eco.lgd * ead
    contrib = expected_contribution_per_loan(p, ead, eco.lgd, eco.annual_rate, eco.funding_rate, eco.opex_per_loan,
                                             eco.horizon_years)
    cm = policy.cost_matrix
    costs = CostAssumptions(mode=cm.mode, missed_default_cost=cm.missed_default_cost,
                            rejected_good_cost=cm.rejected_good_cost, review_cost=cm.review_cost, lgd=eco.lgd,
                            lost_margin_rate=cm.lost_margin_rate)
    c_fn, c_fp = cost_vectors(costs, ead, len(y))
    outcome = evaluate_policy(y, flagged, c_fn, c_fp, cm.review_cost)
    n = len(y)
    tp, fp, fn = outcome["tp"], outcome["fp"], outcome["fn"]
    rules = {r.id: {"action": r.action, "fired": int(res[f"rule:{r.id}"].sum()),
                    "fired_rate": float(res[f"rule:{r.id}"].mean()) if n else 0.0,
                    "evaluable": all(c in frame.columns for c in r.columns())} for r in policy.hard_rules}
    bands = (res.assign(y=y, ead=ead, el=el_12m).groupby("band", observed=True)
             .agg(n=("pd", "size"), avg_pd=("pd", "mean"), observed_default_rate=("y", "mean"), ead=("ead", "sum"),
                  expected_loss=("el", "sum")).reindex(policy.band_labels).fillna(0).reset_index())
    return {
        "policy_version": policy.version,
        "n": n,
        "flag_rate": float(flagged.mean()) if n else 0.0,
        "pd_flag_rate": float(res["pd_flag"].mean()) if n else 0.0,
        "rule_flag_rate": float(res["rule_flag"].mean()) if n else 0.0,
        "flagged_by_rules_only": int((res["rule_flag"] & ~res["pd_flag"]).sum()),
        "confusion": {"tp": tp, "fp": fp, "tn": outcome["tn"], "fn": fn},
        "recall": tp / (tp + fn) if (tp + fn) else None,
        "precision": tp / (tp + fp) if (tp + fp) else None,
        "default_rate_within_threshold": outcome["default_rate_among_approved"],
        "observed_default_rate": float(y.mean()) if n else None,
        "total_ead": float(ead.sum()),
        "ead_within_threshold": float(ead[within].sum()),
        "expected_loss_12m_within_threshold": float(el_12m[within].sum()),
        "expected_loss_horizon_within_threshold": float(el_h[within].sum()),
        "expected_loss_12m_all": float(el_12m.sum()),
        "expected_contribution_within_threshold": float(contrib[within].sum()),
        "realized_contribution_within_threshold": realized_contribution(
            y, within, ead, eco.lgd, eco.annual_rate, eco.funding_rate, eco.opex_per_loan, eco.horizon_years),
        "costs": {k: outcome[k] for k in ("missed_default_cost", "rejected_good_cost", "review_cost_total",
                                          "total_cost", "cost_per_applicant")},
        "rules": rules,
        "bands": bands.to_dict(orient="records"),
        "exposure_imputed": n_imputed,
        "threshold_sources": res["threshold_source"].value_counts().to_dict(),
    }


DELTA_KEYS = ("flag_rate", "recall", "precision", "default_rate_within_threshold", "ead_within_threshold",
              "expected_loss_12m_within_threshold", "expected_contribution_within_threshold",
              "realized_contribution_within_threshold")


def compare(draft: dict[str, Any], active: dict[str, Any]) -> dict[str, float | None]:
    """Draft minus active for the headline metrics (``None`` where either is undefined)."""
    out: dict[str, float | None] = {}
    for k in DELTA_KEYS:
        a, b = draft.get(k), active.get(k)
        out[k] = None if a is None or b is None else float(a) - float(b)
    out["total_cost"] = float(draft["costs"]["total_cost"]) - float(active["costs"]["total_cost"])
    return out
