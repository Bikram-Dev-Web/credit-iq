"""The lending-policy document.

A policy turns a model's PD into an outcome, "within threshold" or "flagged for review" (never
approve/deny), using PD thresholds (a default plus per-segment overrides), hard rules, band cutoffs,
and the cost and economics assumptions used to evaluate it. Policies are versioned and
validated on load; an invalid policy is rejected with an error that names the offending field.
"""
from __future__ import annotations

import hashlib
import json
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from creditiq_core import defaults


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SegmentThreshold(_Strict):
    column: str
    value: str
    threshold: float = Field(gt=0, lt=1)


class PDThreshold(_Strict):
    default: float | Literal["model"] = Field(
        default="model", description="A PD in (0, 1), or 'model' for the model's validation-chosen threshold")
    segments: list[SegmentThreshold] = Field(default_factory=list)

    @field_validator("default")
    @classmethod
    def _range(cls, v: float | str) -> float | str:
        if not isinstance(v, str) and not 0 < v < 1:
            raise ValueError("must be 'model' or a PD strictly between 0 and 1")
        return v


class _Rule(_Strict):
    id: str = Field(pattern=r"^[a-z][a-z0-9_]{1,40}$")
    action: Literal["flag", "note"] = Field(
        default="flag", description="'flag' sends the application for review; 'note' only records the rule")
    reason: str = Field(min_length=3, max_length=200)

    def columns(self) -> list[str]:
        raise NotImplementedError


class RatioLimitRule(_Rule):
    """Fire when numerator / denominator exceeds ``max`` (e.g. annuity-to-income)."""

    type: Literal["ratio_limit"]
    numerator: str
    denominator: str
    max: float = Field(gt=0)

    def columns(self) -> list[str]:
        return [self.numerator, self.denominator]


class MissingValueRule(_Rule):
    """Fire when a column is missing (e.g. no credit-bureau record)."""

    type: Literal["missing_value"]
    column: str

    def columns(self) -> list[str]:
        return [self.column]


class RangeRule(_Rule):
    """Fire when a value is outside [min, max]."""

    type: Literal["range"]
    column: str
    min: float | None = None
    max: float | None = None

    @model_validator(mode="after")
    def _bounds(self) -> RangeRule:
        if self.min is None and self.max is None:
            raise ValueError("a range rule needs min, max or both")
        if self.min is not None and self.max is not None and self.min > self.max:
            raise ValueError("min must not exceed max")
        return self

    def columns(self) -> list[str]:
        return [self.column]


class CategoryInRule(_Rule):
    """Fire when a categorical value is in ``values``."""

    type: Literal["category_in"]
    column: str
    values: list[str] = Field(min_length=1)

    def columns(self) -> list[str]:
        return [self.column]


HardRule = Annotated[RatioLimitRule | MissingValueRule | RangeRule | CategoryInRule, Field(discriminator="type")]


class CostMatrix(_Strict):
    """Asymmetric cost of decisions (see ``creditiq_core.finance.cost_matrix``)."""

    mode: Literal["per_loan", "fixed"] = "per_loan"
    missed_default_cost: float = Field(default=200_000.0, ge=0)
    rejected_good_cost: float = Field(default=40_000.0, ge=0)
    review_cost: float = Field(default=500.0, ge=0)
    lost_margin_rate: float = Field(default=0.08, ge=0, le=1)


class Economics(_Strict):
    lgd: float = Field(default=defaults.DEFAULT_LGD, ge=0, le=1)
    ccf: float = Field(default=defaults.DEFAULT_CCF, gt=0, le=2)
    annual_rate: float = Field(default=defaults.DEFAULT_ANNUAL_RATE, ge=0, le=2)
    funding_rate: float = Field(default=defaults.DEFAULT_FUNDING_RATE, ge=0, le=2)
    opex_per_loan: float = Field(default=defaults.DEFAULT_OPEX_PER_LOAN, ge=0)
    horizon_years: float = Field(default=1.0, gt=0, le=30)
    default_exposure: float = Field(default=100_000.0, gt=0, description="EAD used when the exposure is missing")


class Policy(_Strict):
    version: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,40}$")
    name: str = Field(min_length=3, max_length=120)
    description: str = ""
    parent_version: str | None = None
    pd_threshold: PDThreshold = Field(default_factory=PDThreshold)
    band_cutoffs: list[float] = Field(default_factory=lambda: list(defaults.DEFAULT_BAND_CUTOFFS))
    band_labels: list[str] = Field(default_factory=lambda: list(defaults.BAND_LABELS))
    hard_rules: list[HardRule] = Field(default_factory=list)
    cost_matrix: CostMatrix = Field(default_factory=CostMatrix)
    economics: Economics = Field(default_factory=Economics)

    @model_validator(mode="after")
    def _consistency(self) -> Policy:
        cut = self.band_cutoffs
        if not cut or sorted(cut) != cut or len(set(cut)) != len(cut) or any(c <= 0 or c >= 1 for c in cut):
            raise ValueError("band_cutoffs must be strictly increasing PDs between 0 and 1")
        if len(self.band_labels) != len(cut) + 1:
            raise ValueError(f"band_labels needs {len(cut) + 1} labels for {len(cut)} cutoffs")
        ids = [r.id for r in self.hard_rules]
        dupes = sorted({i for i in ids if ids.count(i) > 1})
        if dupes:
            raise ValueError(f"hard rule ids must be unique (repeated: {', '.join(dupes)})")
        seen: set[tuple[str, str]] = set()
        for s in self.pd_threshold.segments:
            if (s.column, s.value) in seen:
                raise ValueError(f"segment threshold for {s.column} = {s.value!r} is defined twice")
            seen.add((s.column, s.value))
        return self

    def referenced_columns(self) -> set[str]:
        cols = {s.column for s in self.pd_threshold.segments}
        for r in self.hard_rules:
            cols.update(r.columns())
        return cols

    def content_hash(self) -> str:
        return hashlib.sha256(json.dumps(self.model_dump(mode="json"), sort_keys=True).encode()).hexdigest()[:16]


class PolicyValidationError(ValueError):
    """A policy document that does not validate. ``errors`` holds one readable line per problem."""

    def __init__(self, errors: list[str]):
        self.errors = errors
        super().__init__("Invalid policy:\n" + "\n".join(f"  - {e}" for e in errors))


def _format(err: dict[str, Any]) -> str:
    loc = ".".join(str(p) for p in err.get("loc", ()) if not (isinstance(p, str) and p in {
        "ratio_limit", "missing_value", "range", "category_in"}))
    return f"{loc or 'policy'}: {err.get('msg', 'invalid')}"


def parse_policy(data: dict[str, Any], *, model_columns: set[str] | None = None,
                 sensitive_columns: set[str] | None = None) -> Policy:
    """Validate a policy document. With ``model_columns``, every column a rule or segment references must be a
    model input; sensitive attributes may never be used by a policy."""
    try:
        policy = Policy.model_validate(data)
    except ValidationError as exc:
        raise PolicyValidationError([_format(e) for e in exc.errors()]) from exc
    errors = []
    for col in sorted(policy.referenced_columns()):
        if sensitive_columns and col in sensitive_columns:
            errors.append(f"{col}: sensitive attributes cannot be used in lending rules (audit only)")
        elif model_columns is not None and col not in model_columns:
            errors.append(f"{col}: not an input of the active model, so the rule could never be evaluated")
    if errors:
        raise PolicyValidationError(errors)
    return policy
