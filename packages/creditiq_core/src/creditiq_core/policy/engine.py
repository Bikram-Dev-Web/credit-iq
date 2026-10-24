"""Apply a policy to model output: thresholds, hard rules and the resulting outcome.

The same vectorised code path serves a single decision (one row) and a portfolio simulation
(thousands of rows), so a simulated policy behaves exactly like the live one.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from creditiq_core.finance.portfolio import assign_risk_band
from creditiq_core.policy.schema import CategoryInRule, MissingValueRule, Policy, RangeRule, RatioLimitRule

WITHIN = "within_threshold"
FLAGGED = "flagged_for_review"


@dataclass
class RuleResult:
    id: str
    type: str
    action: str
    fired: bool
    reason: str
    value: float | str | None = None
    limit: float | str | list[str] | None = None
    evaluable: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "type": self.type, "action": self.action, "fired": self.fired, "reason": self.reason,
                "value": self.value, "limit": self.limit, "evaluable": self.evaluable}


@dataclass
class Decision:
    outcome: str  # WITHIN | FLAGGED
    pd: float
    threshold: float
    threshold_source: str  # "model" | "policy default" | "segment <column>=<value>"
    model_threshold: float
    band: str
    rules: list[RuleResult] = field(default_factory=list)

    @property
    def flagged(self) -> bool:
        return self.outcome == FLAGGED

    @property
    def rules_fired(self) -> list[RuleResult]:
        return [r for r in self.rules if r.fired]

    def reasons(self) -> list[str]:
        out = []
        if self.pd >= self.threshold:
            out.append(f"PD {self.pd:.1%} is at or above the {self.threshold_source} threshold {self.threshold:.1%}")
        out += [f"{r.id}: {r.reason}" for r in self.rules_fired if r.action == "flag"]
        return out


def _num(frame: pd.DataFrame, col: str) -> np.ndarray:
    if col not in frame.columns:
        return np.full(len(frame), np.nan)
    return np.asarray(pd.to_numeric(frame[col], errors="coerce"), dtype=float)


def thresholds(policy: Policy, frame: pd.DataFrame, model_threshold: float) -> tuple[np.ndarray, list[str]]:
    """Per-row threshold and its source. The last matching segment wins (documented order of precedence)."""
    base = model_threshold if policy.pd_threshold.default == "model" else float(policy.pd_threshold.default)
    src = "model" if policy.pd_threshold.default == "model" else "policy default"
    thr = np.full(len(frame), base, dtype=float)
    source = np.full(len(frame), src, dtype=object)
    for seg in policy.pd_threshold.segments:
        if seg.column not in frame.columns:
            continue
        m = (frame[seg.column].astype(str) == seg.value).to_numpy()
        thr[m] = seg.threshold
        source[m] = f"segment {seg.column}={seg.value}"
    return thr, [str(s) for s in source]


def evaluate_rules(policy: Policy, frame: pd.DataFrame) -> dict[str, np.ndarray]:
    """Boolean "fired" vector per rule. A rule whose columns are absent never fires (reported as not evaluable)."""
    fired: dict[str, np.ndarray] = {}
    for rule in policy.hard_rules:
        if isinstance(rule, RatioLimitRule):
            num, den = _num(frame, rule.numerator), _num(frame, rule.denominator)
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = np.where(den > 0, num / den, np.nan)
            fired[rule.id] = np.nan_to_num(ratio, nan=-np.inf) > rule.max
        elif isinstance(rule, MissingValueRule):
            fired[rule.id] = (frame[rule.column].isna().to_numpy() if rule.column in frame.columns
                              else np.zeros(len(frame), dtype=bool))
        elif isinstance(rule, RangeRule):
            v = _num(frame, rule.column)
            lo = v < rule.min if rule.min is not None else np.zeros(len(v), dtype=bool)
            hi = v > rule.max if rule.max is not None else np.zeros(len(v), dtype=bool)
            fired[rule.id] = (lo | hi) & ~np.isnan(v)
        elif isinstance(rule, CategoryInRule):
            fired[rule.id] = (frame[rule.column].astype(str).isin(rule.values).to_numpy()
                              if rule.column in frame.columns else np.zeros(len(frame), dtype=bool))
    return fired


def apply_policy(policy: Policy, frame: pd.DataFrame, pd_values: np.ndarray,
                 model_threshold: float) -> pd.DataFrame:
    """Vectorised outcome per row: columns ``threshold``, ``threshold_source``, ``band``, ``pd_flag``,
    ``rule_flag``, ``flagged`` and one boolean column per rule (``rule:<id>``)."""
    p = np.asarray(pd_values, dtype=float)
    thr, source = thresholds(policy, frame, model_threshold)
    fired = evaluate_rules(policy, frame)
    out = pd.DataFrame({"pd": p, "threshold": thr, "threshold_source": source,
                        "band": assign_risk_band(p, policy.band_cutoffs, policy.band_labels).astype(str).to_numpy()})
    out["pd_flag"] = p >= thr
    rule_flag = np.zeros(len(p), dtype=bool)
    for rule in policy.hard_rules:
        out[f"rule:{rule.id}"] = fired[rule.id]
        if rule.action == "flag":
            rule_flag |= fired[rule.id]
    out["rule_flag"] = rule_flag
    out["flagged"] = out["pd_flag"] | out["rule_flag"]
    return out


def decide(policy: Policy, applicant: dict[str, Any], pd_value: float, model_threshold: float) -> Decision:
    """Outcome for one applicant, with every rule's result (fired or not) for the audit trail."""
    frame = pd.DataFrame([applicant])
    row = apply_policy(policy, frame, np.array([pd_value]), model_threshold).iloc[0]
    results = []
    for rule in policy.hard_rules:
        cols = rule.columns()
        evaluable = all(c in applicant for c in cols)
        value: float | str | None = None
        limit: float | str | list[str] | None = None
        if isinstance(rule, RatioLimitRule):
            num, den = _num(frame, rule.numerator)[0], _num(frame, rule.denominator)[0]
            value = float(num / den) if np.isfinite(num) and np.isfinite(den) and den > 0 else None
            limit = rule.max
        elif isinstance(rule, MissingValueRule):
            value = "missing" if bool(row[f"rule:{rule.id}"]) else "present"
        elif isinstance(rule, RangeRule):
            v = _num(frame, rule.column)[0]
            value, limit = (float(v) if np.isfinite(v) else None), f"[{rule.min}, {rule.max}]"
        elif isinstance(rule, CategoryInRule):
            value, limit = (str(applicant.get(rule.column)) if applicant.get(rule.column) is not None else None,
                            list(rule.values))
        results.append(RuleResult(rule.id, rule.type, rule.action, bool(row[f"rule:{rule.id}"]), rule.reason,
                                  value, limit, evaluable))
    return Decision(outcome=FLAGGED if bool(row["flagged"]) else WITHIN, pd=float(pd_value),
                    threshold=float(row["threshold"]), threshold_source=str(row["threshold_source"]),
                    model_threshold=float(model_threshold), band=str(row["band"]), rules=results)
