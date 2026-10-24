"""Lending policies: list, read, validate, simulate, compare, save and activate."""
from __future__ import annotations

from typing import Any

import yaml
from fastapi import APIRouter, Depends, HTTPException, Request, status

from creditiq_api import deps
from creditiq_api.model_store import LoadedModel
from creditiq_api.policy_store import PolicyConflict, PolicyStore
from creditiq_api.schemas import (
    COMMON_ERRORS,
    CompareRequest,
    PolicyDocument,
    PolicyInput,
    PolicySummary,
    ResultOut,
    SimulateRequest,
    SimulationOut,
    ValidationResult,
)
from creditiq_api.scoring import data_provenance, labels
from creditiq_api.security import Principal, admin, analyst
from creditiq_core.finance.cost_matrix import CostAssumptions, compare_policies
from creditiq_core.policy.schema import Policy, PolicyValidationError, parse_policy
from creditiq_core.policy.simulate import compare, simulate
from creditiq_core.provenance import Provenance as P

router = APIRouter(prefix="/v1/policies", tags=["policies"], responses=COMMON_ERRORS)


def _parse(body: PolicyInput, model: LoadedModel | None) -> Policy:
    if (body.policy is None) == (body.yaml is None):
        raise PolicyValidationError(["send exactly one of `policy` (JSON) or `yaml`"])
    data: Any = body.policy
    if body.yaml is not None:
        try:
            data = yaml.safe_load(body.yaml)
        except yaml.YAMLError as exc:
            raise PolicyValidationError([f"yaml: not valid YAML ({exc})"]) from exc
        if not isinstance(data, dict):
            raise PolicyValidationError(["yaml: the document must be a mapping"])
    return parse_policy(data, model_columns=set(model.input_columns) if model else None,
                        sensitive_columns=set(model.snapshot.sensitive_cols) if model else None)


def _invalid(exc: PolicyValidationError) -> HTTPException:
    return HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, {"message": "Invalid policy", "errors": exc.errors})


@router.get("", response_model=list[PolicySummary], summary="All policy versions")
def list_policies(_: Principal = Depends(analyst), store: PolicyStore = Depends(deps.policies)) -> list[dict[str, Any]]:
    return store.list()


@router.get("/active", response_model=PolicyDocument, summary="The active policy")
def active_policy(_: Principal = Depends(analyst), store: PolicyStore = Depends(deps.policies),
                  policy: Policy = Depends(deps.policy)) -> dict[str, Any]:
    _p, summary = store.get(policy.version)
    return {"summary": summary, "policy": policy.model_dump(mode="json"), "yaml": store.to_yaml(policy)}


@router.get("/{version}", response_model=PolicyDocument, summary="One policy version")
def get_policy(version: str, _: Principal = Depends(analyst), store: PolicyStore = Depends(deps.policies)) -> dict[str, Any]:
    try:
        policy, summary = store.get(version)
    except KeyError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Policy {version} not found.") from exc
    return {"summary": summary, "policy": policy.model_dump(mode="json"), "yaml": store.to_yaml(policy)}


@router.post("/validate", response_model=ValidationResult, summary="Validate a draft policy (JSON or YAML)")
def validate(body: PolicyInput, request: Request, _: Principal = Depends(analyst)) -> dict[str, Any]:
    """Always 200: `valid` and readable `errors` (field path: problem). Rules may only reference inputs of the served
    model and never sensitive attributes."""
    try:
        policy = _parse(body, request.app.state.holder.current)
    except PolicyValidationError as exc:
        return {"valid": False, "errors": exc.errors, "policy": None}
    return {"valid": True, "errors": [], "policy": policy.model_dump(mode="json")}


@router.post("/simulate", response_model=SimulationOut, summary="Simulate a draft policy against the active one")
def simulate_policy(body: SimulateRequest, _: Principal = Depends(analyst), model: LoadedModel = Depends(deps.model),
                    active: Policy = Depends(deps.policy)) -> dict[str, Any]:
    """Runs the draft and the active policy over held-out applicants and returns flag rate, expected loss,
    expected contribution, confusion matrix, costs and the change versus the active policy.

    Default split: **validation**, where choices are made. `split=test` is a read-only final check: iterating on
    test results would turn the test set into a tuning set and make its numbers optimistic."""
    try:
        draft = _parse(body, model)
    except PolicyValidationError as exc:
        raise _invalid(exc) from exc
    frame = model.snapshot.split(body.split)
    d = simulate(draft, frame, model.artifact.threshold, model.exposure_col)
    a = simulate(active, frame, model.artifact.threshold, model.exposure_col)
    warnings = []
    if body.split == "test":
        warnings.append("TEST split: use this as a final check only. Do not tune the policy on these numbers; "
                        "go back to the validation split to iterate.")
    for r, info in d["rules"].items():
        if not info["evaluable"]:
            warnings.append(f"Rule {r} references a column that is not in the evaluation data; it never fires here.")
    if model.is_synthetic:
        warnings.append("SYNTHETIC data: these figures describe fabricated applicants.")
    obs, meas = labels(model, P.OBSERVED), labels(model, P.MEASURED)
    sim = labels(model, P.SIMULATED, P.ASSUMPTION)
    prov = {k: meas for k in ("flag_rate", "pd_flag_rate", "rule_flag_rate", "recall", "precision", "tp", "fp", "tn",
                              "fn", "fired", "fired_rate", "flagged_by_rules_only", "default_rate_within_threshold")}
    prov |= {k: sim for k in ("expected_loss_12m_within_threshold", "expected_loss_horizon_within_threshold",
                              "expected_loss_12m_all", "expected_contribution_within_threshold",
                              "realized_contribution_within_threshold", "missed_default_cost", "rejected_good_cost",
                              "review_cost_total", "total_cost", "cost_per_applicant", "expected_loss")}
    prov |= {"n": obs, "observed_default_rate": obs, "total_ead": labels(model, P.OBSERVED, P.ASSUMPTION),
             "ead_within_threshold": labels(model, P.OBSERVED, P.ASSUMPTION), "ead": labels(model, P.OBSERVED,
                                                                                            P.ASSUMPTION),
             "avg_pd": labels(model, P.MODEL), "exposure_imputed": obs}
    return {"split": body.split, "model_version": model.version, "draft": d, "active": a, "delta": compare(d, a),
            "warnings": warnings, "data_provenance": data_provenance(model), "provenance": prov}


@router.post("/compare", response_model=ResultOut, summary="Benchmark policies (v1): approve-all, fixed, "
                                                            "cost-minimising, expected-contribution")
def compare_benchmarks(body: CompareRequest, _: Principal = Depends(analyst), model: LoadedModel = Depends(deps.model),
                       policy: Policy = Depends(deps.policy)) -> dict[str, Any]:
    """Evaluated on TEST; the cost-minimising threshold is chosen on VALIDATION (v1's policy comparison)."""
    from creditiq_core.policy.simulate import exposure

    val, test = model.snapshot.split("validation"), model.snapshot.split("test")
    ead_v, _imputed_v = exposure(val, model.exposure_col, policy)
    ead_t, _imputed_t = exposure(test, model.exposure_col, policy)
    cm, eco = policy.cost_matrix, policy.economics
    costs = CostAssumptions(mode=cm.mode, missed_default_cost=cm.missed_default_cost,
                            rejected_good_cost=cm.rejected_good_cost, review_cost=cm.review_cost, lgd=eco.lgd,
                            lost_margin_rate=cm.lost_margin_rate)
    table, info = compare_policies(val["y"], val["pd"], ead_v, test["y"], test["pd"], ead_t,
                                   body.fixed_threshold or model.artifact.threshold, costs, eco.annual_rate,
                                   eco.funding_rate, eco.opex_per_loan, eco.horizon_years)
    table = table.rename(columns={"approval_rate": "within_threshold_rate",
                                  "default_rate_among_approved": "default_rate_within_threshold",
                                  "realized_contribution_approved_book": "realized_contribution_within_threshold"})
    table["policy"] = table["policy"].str.replace("approve all", "nobody flagged", regex=False).str.replace(
        "approve if E[contribution] > 0", "within threshold if E[contribution] > 0", regex=False)
    from creditiq_api.routers.analytics import _records

    meas, sim = labels(model, P.MEASURED), labels(model, P.SIMULATED, P.ASSUMPTION)
    prov = {k: meas for k in ("tp", "fp", "tn", "fn", "flagged", "within_threshold_rate",
                              "default_rate_within_threshold")}
    prov |= {k: sim for k in ("missed_default_cost", "rejected_good_cost", "review_cost_total", "total_cost",
                              "cost_per_applicant", "realized_contribution_within_threshold")}
    prov["cost_optimal_threshold"] = labels(model, P.MEASURED, P.ASSUMPTION)
    return {"result": {"policies": _records(table), **info, "evaluated_on": "test",
                       "cost_threshold_chosen_on": "validation"},
            "data_provenance": data_provenance(model), "provenance": prov}


@router.post("", response_model=PolicySummary, status_code=status.HTTP_201_CREATED,
             summary="Save a new policy version (admin)")
def create_policy(body: PolicyInput, request: Request, principal: Principal = Depends(admin),
                  store: PolicyStore = Depends(deps.policies)) -> dict[str, Any]:
    """Versions are immutable: saving an existing version number is a 409. Export the YAML (GET) to commit it to
    `policies/` for review."""
    try:
        policy = _parse(body, request.app.state.holder.current)
    except PolicyValidationError as exc:
        raise _invalid(exc) from exc
    try:
        return store.create(policy, principal.key_id)
    except PolicyConflict as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from exc


@router.post("/{version}/activate", response_model=ResultOut, summary="Make a policy version active (admin)")
def activate(version: str, request: Request, principal: Principal = Depends(admin),
             store: PolicyStore = Depends(deps.policies)) -> dict[str, Any]:
    try:
        res = store.activate(version, principal.key_id)
    except KeyError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Policy {version} not found.") from exc
    request.app.state.metrics.set_policy(version)
    return {"result": res, "data_provenance": "observed", "provenance": {}}
