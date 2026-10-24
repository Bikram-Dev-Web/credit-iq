"""Request and response models (Pydantic v2).

Provenance: every response carries ``provenance``, a map from field name to the labels describing where
that number comes from (observed, measured, model, assumption, simulated, synthetic), plus
``data_provenance`` ("synthetic" or "observed") for the data the model and analytics are built on.
Keys are field names wherever the field appears (top level or inside rows).
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from creditiq_core.defaults import APP_DISCLAIMER
from creditiq_core.provenance import Provenance

ProvenanceMap = dict[str, list[Provenance]]
Outcome = Literal["within_threshold", "flagged_for_review"]
Split = Literal["validation", "test"]


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class WithProvenance(BaseModel):
    data_provenance: Literal["synthetic", "observed"]
    provenance: ProvenanceMap
    disclaimer: str = APP_DISCLAIMER


# ---------------------------------------------------------------------------- scoring
class ScoreRequest(Strict):
    applicant: dict[str, Any] = Field(
        description="Every model input (see GET /v1/models/{version}/schema). Use null for unknown values; "
                    "they are imputed from the training data.")
    exposure: float | None = Field(default=None, ge=0, description="Exposure override; default: the model's "
                                                                   "exposure input (e.g. AMT_CREDIT)")


class Reason(BaseModel):
    feature: str
    value: str
    contribution: float
    direction: Literal["increases_risk", "decreases_risk"]
    derived: bool = Field(description="True for an engineered ratio built from the inputs")


class Uncertainty(BaseModel):
    test_roc_auc: float | None
    test_calibration_error: float | None = Field(description="Expected calibration error on the held-out test split")
    band_observed_default_rate: float | None = Field(
        description="Observed default rate of held-out test applicants in the same risk band")
    band_mean_pd: float | None
    band_n: int


class RuleOut(BaseModel):
    id: str
    type: str
    action: Literal["flag", "note"]
    fired: bool
    reason: str
    value: float | str | None = None
    limit: float | str | list[str] | None = None
    evaluable: bool = True


class ScoreResponse(WithProvenance):
    request_id: str
    model_name: str
    model_version: str
    policy_version: str
    pd: float = Field(ge=0, le=1)
    raw_score: float
    is_calibrated: bool
    risk_band: str
    threshold: float
    threshold_source: str
    flagged_for_review: bool
    outcome: Outcome
    ead: float
    lgd: float
    expected_loss: float = Field(description="12-month expected loss: PD × LGD × EAD")
    top_reasons: list[Reason]
    explanation_method: str
    explanation_units: str
    uncertainty: Uncertainty
    warnings: list[str]
    caveats: list[str]
    latency_ms: float


class DecideResponse(ScoreResponse):
    policy_name: str
    rules: list[RuleOut]
    rules_fired: list[str]
    reasons: list[str]


class BatchScoreRequest(Strict):
    applicants: list[dict[str, Any]] = Field(min_length=1)


# ---------------------------------------------------------------------------- models
class ModelVersionOut(BaseModel):
    version: str
    aliases: list[str]
    created: int | None
    description: str | None
    run_id: str | None
    is_synthetic: bool | None = None
    dataset: str | None = None
    selected_model: str | None = None
    test_roc_auc: float | None = None
    test_pr_auc: float | None = None
    test_brier: float | None = None
    serving: bool = False


class ModelList(WithProvenance):
    model_name: str
    serving_alias: str
    serving_version: str | None
    versions: list[ModelVersionOut]


class PromoteRequest(Strict):
    alias: Literal["production", "staging"] = "production"


class PromoteResponse(BaseModel):
    model_name: str
    alias: str
    version: str
    replaced: str | None
    serving_version: str | None


class FieldSpec(BaseModel):
    name: str
    kind: Literal["numeric", "categorical"]
    importance_rank: int
    importance: float | None
    missing_rate: float
    min: float | None = None
    max: float | None = None
    p01: float | None = None
    p99: float | None = None
    median: float | None = None
    is_integer: bool | None = None
    categories: list[str] | None = None
    mode: str | None = None
    advanced: bool = Field(description="Shown in the 'advanced' section of the scoring form")


class ModelSchemaOut(WithProvenance):
    model_version: str
    exposure_field: str | None
    fields: list[FieldSpec]
    json_schema: dict[str, Any]


# ---------------------------------------------------------------------------- policies
class PolicySummary(BaseModel):
    version: str
    name: str
    source: str
    created_at: datetime
    created_by: str
    is_active: bool
    content_hash: str
    parent_version: str | None
    description: str


class PolicyDocument(BaseModel):
    summary: PolicySummary
    policy: dict[str, Any]
    yaml: str


class PolicyInput(Strict):
    policy: dict[str, Any] | None = Field(default=None, description="Policy as JSON")
    yaml: str | None = Field(default=None, description="Policy as YAML (alternative to `policy`)")


class ValidationResult(BaseModel):
    valid: bool
    errors: list[str]
    policy: dict[str, Any] | None = None


class SimulateRequest(PolicyInput):
    split: Split = Field(default="validation", description="validation (default) or test (read-only final check)")


class SimulationOut(WithProvenance):
    split: Split
    model_version: str
    draft: dict[str, Any]
    active: dict[str, Any]
    delta: dict[str, float | None]
    warnings: list[str]


class CompareRequest(Strict):
    fixed_threshold: float | None = Field(default=None, gt=0, lt=1)


# ---------------------------------------------------------------------------- analytics
class Assumptions(Strict):
    lgd: float = Field(ge=0, le=1)
    ccf: float = Field(gt=0, le=2)


class TableOut(WithProvenance):
    model_version: str
    split: Split
    assumptions: dict[str, float]
    rows: list[dict[str, Any]]
    summary: dict[str, Any] = Field(default_factory=dict)


class EconomicsRequest(Strict):
    n_loans: int = Field(default=1000, ge=0, le=10_000_000)
    avg_exposure: float = Field(default=500_000.0, ge=0)
    pd: float = Field(default=0.08, ge=0, le=1)
    lgd: float = Field(default=0.45, ge=0, le=1)
    annual_rate: float = Field(default=0.16, ge=0, le=2)
    funding_rate: float = Field(default=0.07, ge=0, le=2)
    opex_per_loan: float = Field(default=1500.0, ge=0)
    horizon_years: float = Field(default=1.0, gt=0, le=30)
    upfront_fee_rate: float = Field(default=0.0, ge=0, le=2)
    interest_recovery_on_default: float = Field(default=0.0, ge=0, le=1)
    sensitivity_shock: float = Field(default=0.2, gt=0, le=1)


class Scenario(Strict):
    scenario: str = Field(min_length=1, max_length=60)
    pd_multiplier: float = Field(ge=0, le=20)
    lgd_shift: float = Field(ge=-1, le=1)
    ead_multiplier: float = Field(ge=0, le=20)


class StressRequest(Strict):
    scenarios: list[Scenario] | None = Field(default=None, max_length=20, description="Default: the four v1 scenarios")
    split: Split = "test"
    lgd: float | None = Field(default=None, ge=0, le=1)
    ccf: float | None = Field(default=None, gt=0, le=2)
    annual_rate: float | None = Field(default=None, ge=0, le=2)
    funding_rate: float | None = Field(default=None, ge=0, le=2)
    opex_per_loan: float | None = Field(default=None, ge=0)
    horizon_years: float | None = Field(default=None, gt=0, le=30)


class NPVRequest(Strict):
    principal: float = Field(default=500_000.0, gt=0, le=1e12)
    annual_rate: float = Field(default=0.16, ge=0, le=2)
    term_months: int = Field(default=36, ge=1, le=600)
    annual_pd: float = Field(default=0.08, ge=0, le=1)
    lgd: float = Field(default=0.45, ge=0, le=1)
    discount_rate: float = Field(default=0.07, ge=0, le=2)
    recovery_lag_months: int = Field(default=6, ge=0, le=120)
    upfront_fee: float = Field(default=0.0, ge=0)
    servicing_cost_monthly: float = Field(default=0.0, ge=0)


class ResultOut(WithProvenance):
    result: Any


class MetaOut(WithProvenance):
    api_version: str
    environment: str
    role: Literal["analyst", "admin"]
    admin_actions_enabled: bool
    model_name: str
    model_version: str | None
    model_label: str | None
    model_loaded_at: datetime | None
    policy_version: str | None
    policy_name: str | None
    grafana_url: str | None
    mlflow_ui_url: str | None
    provenance_labels: dict[str, list[str]]


# ---------------------------------------------------------------------------- errors
class ErrorBody(BaseModel):
    message: str
    errors: list[Any] | None = None


class ErrorOut(BaseModel):
    error: ErrorBody | dict[str, Any]
    request_id: str | None = None
    detail: list[Any] | None = None


COMMON_ERRORS: dict[int | str, dict[str, Any]] = {
    401: {"model": ErrorOut, "description": "Missing or invalid API key"},
    403: {"model": ErrorOut, "description": "Role not allowed, or admin actions disabled on this deployment"},
    404: {"model": ErrorOut, "description": "Not found"},
    413: {"model": ErrorOut, "description": "Payload too large"},
    415: {"model": ErrorOut, "description": "Unsupported media type"},
    409: {"model": ErrorOut, "description": "Conflict (immutable version already exists)"},
    422: {"model": ErrorOut, "description": "Validation error (field location and message)"},
    429: {"model": ErrorOut, "description": "Rate limit exceeded"},
    503: {"model": ErrorOut, "description": "No model served, no active policy, or the audit store is down"},
}
