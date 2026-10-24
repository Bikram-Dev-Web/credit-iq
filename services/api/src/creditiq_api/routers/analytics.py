"""Portfolio, scenario and fairness analytics on the served model's held-out evaluation snapshot.

The portfolio defaults to the TEST split (out-of-sample, reported not tuned). Observed default rate (labels),
average PD (model) and expected loss (model × assumptions) are kept as separate numbers throughout.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query, status

from creditiq_api import deps
from creditiq_api.model_store import LoadedModel
from creditiq_api.schemas import COMMON_ERRORS, EconomicsRequest, NPVRequest, ResultOut, StressRequest, TableOut
from creditiq_api.scoring import data_provenance, labels
from creditiq_api.security import Principal, analyst
from creditiq_core import fairness as fair
from creditiq_core.finance import economics as fc
from creditiq_core.finance import portfolio as pa
from creditiq_core.jsonsafe import sanitize
from creditiq_core.policy.schema import Policy
from creditiq_core.provenance import Provenance as P

router = APIRouter(prefix="/v1", tags=["analytics"], responses=COMMON_ERRORS)

Split = Query("test", pattern="^(validation|test)$", description="Held-out split (default: test)")
LGD = Query(None, ge=0, le=1, description="LGD assumption (default: the active policy's)")
CCF = Query(None, gt=0, le=2, description="CCF assumption (default: the active policy's)")


def _records(df: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = sanitize(df.astype(object).where(df.notna(), None).to_dict(orient="records"))
    return [{str(k): v for k, v in r.items()} for r in rows]


def _portfolio(model: LoadedModel, policy: Policy, split: str, lgd: float | None,
               ccf: float | None) -> tuple[pd.DataFrame, dict[str, float]]:
    frame = model.snapshot.split(split)
    a = {"lgd": policy.economics.lgd if lgd is None else lgd, "ccf": policy.economics.ccf if ccf is None else ccf}
    port = pa.build_portfolio(frame.drop(columns=["split", "y", "pd", "raw_score"]), frame["y"], frame["pd"],
                              model.exposure_col, a["lgd"], a["ccf"], policy.economics.default_exposure,
                              policy.band_cutoffs)
    port["risk_band"] = pa.assign_risk_band(port["pd"], policy.band_cutoffs, policy.band_labels).to_numpy()
    return port, a


def _prov(model: LoadedModel) -> dict[str, list[P]]:
    obs, mod = labels(model, P.OBSERVED), labels(model, P.MODEL)
    sim = labels(model, P.MODEL, P.ASSUMPTION)
    return {
        "n_loans": obs, "loans": obs, "total_ead": labels(model, P.OBSERVED, P.ASSUMPTION),
        "ead": labels(model, P.OBSERVED, P.ASSUMPTION),
        "observed_default_rate": obs, "observed_default_rate_ead_weighted": obs, "avg_pd": mod,
        "ead_weighted_pd": mod, "total_el": sim, "expected_loss": sim, "el_rate": sim, "el_share": sim,
        "high_risk_share_count": mod, "high_risk_share_ead": mod, "loan_share": obs, "ead_share": obs,
        "hhi": obs, "top1_share": obs, "top3_share": obs, "effective_segments": obs,
        "share_of_loans": sim, "share_of_expected_loss": sim, "el_share_riskiest_10pct": sim, "decile": mod, "lgd": [P.ASSUMPTION],
        "ccf": [P.ASSUMPTION], "exposure_imputed": obs,
    }


def _table(model: LoadedModel, split: str, a: dict[str, float], rows: list[dict[str, Any]],
           summary: dict[str, Any] | None = None) -> dict[str, Any]:
    return {"model_version": model.version, "split": split, "assumptions": a, "rows": rows,
            "summary": sanitize(summary or {}), "data_provenance": data_provenance(model), "provenance": _prov(model)}


@router.get("/portfolio/summary", response_model=TableOut, tags=["portfolio"], summary="Portfolio KPIs")
def portfolio_summary(split: str = Split, lgd: float | None = LGD, ccf: float | None = CCF,
                      _: Principal = Depends(analyst), model: LoadedModel = Depends(deps.model),
                      policy: Policy = Depends(deps.policy)) -> dict[str, Any]:
    port, a = _portfolio(model, policy, split, lgd, ccf)
    kpis = pa.portfolio_kpis(port, high_risk_labels=tuple(policy.band_labels[-2:]))
    kpis["exposure_imputed"] = int(port["exposure_imputed"].sum())
    return _table(model, split, a, [], kpis)


@router.get("/portfolio/bands", response_model=TableOut, tags=["portfolio"], summary="Exposure and EL by risk band")
def portfolio_bands(split: str = Split, lgd: float | None = LGD, ccf: float | None = CCF,
                    _: Principal = Depends(analyst), model: LoadedModel = Depends(deps.model),
                    policy: Policy = Depends(deps.policy)) -> dict[str, Any]:
    port, a = _portfolio(model, policy, split, lgd, ccf)
    port["risk_band"] = pd.Categorical(port["risk_band"], categories=policy.band_labels, ordered=True)
    table = pa.band_summary(port)
    return _table(model, split, a, _records(table), {"band_definitions": pa.band_description(policy.band_cutoffs,
                                                                                           policy.band_labels)})


@router.get("/portfolio/deciles", response_model=TableOut, tags=["portfolio"],
            summary="Observed default rate vs average PD by PD decile")
def portfolio_deciles(split: str = Split, lgd: float | None = LGD, ccf: float | None = CCF,
                      _: Principal = Depends(analyst), model: LoadedModel = Depends(deps.model),
                      policy: Policy = Depends(deps.policy)) -> dict[str, Any]:
    port, a = _portfolio(model, policy, split, lgd, ccf)
    return _table(model, split, a, _records(pa.pd_decile_view(port)))


@router.get("/portfolio/segments", response_model=TableOut, tags=["portfolio"], summary="EL and exposure by segment")
def portfolio_segments(column: str | None = Query(None, description="Segment column (default: the first "
                                                                     "categorical model input)"),
                       split: str = Split, lgd: float | None = LGD, ccf: float | None = CCF,
                       _: Principal = Depends(analyst), model: LoadedModel = Depends(deps.model),
                       policy: Policy = Depends(deps.policy)) -> dict[str, Any]:
    port, a = _portfolio(model, policy, split, lgd, ccf)
    allowed = [c for c in model.snapshot.predictors if c in port.columns]
    if column is None:
        column = next((c for c in allowed if model.artifact.input_schema.get(c, {}).get("kind") == "categorical"),
                      allowed[0] if allowed else None)
    if column not in allowed:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT,
                            f"Segment column must be a model input; choose one of: {', '.join(allowed)}")
    table = pa.segment_summary(port, column).rename(columns={column: "segment"})
    return _table(model, split, a, _records(table),
                  {"column": column, "available_columns": allowed, **pa.concentration_metrics(table)})


@router.get("/portfolio/concentration", response_model=TableOut, tags=["portfolio"],
            summary="Loss-concentration curve and HHI")
def portfolio_concentration(column: str | None = None, split: str = Split, lgd: float | None = LGD,
                            ccf: float | None = CCF, _: Principal = Depends(analyst),
                            model: LoadedModel = Depends(deps.model),
                            policy: Policy = Depends(deps.policy)) -> dict[str, Any]:
    port, a = _portfolio(model, policy, split, lgd, ccf)
    curve = pa.loss_concentration_curve(port, points=51)
    summary: dict[str, Any] = {}
    allowed = [c for c in model.snapshot.predictors if c in port.columns]
    col = column or next((c for c in allowed if model.artifact.input_schema.get(c, {}).get("kind") == "categorical"),
                         None)
    if col is not None:
        if col not in allowed:
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, f"Unknown segment column '{col}'.")
        summary = {"column": col, **pa.concentration_metrics(pa.segment_summary(port, col))}
    el = np.sort(port["expected_loss"].to_numpy())[::-1]
    if el.sum() > 0:
        top10 = float(el[: max(int(round(len(el) * 0.1)), 1)].sum()) / float(el.sum())
        summary["el_share_riskiest_10pct"] = float(top10)
    return _table(model, split, a, _records(curve), summary)


# ---------------------------------------------------------------------------- scenarios

@router.post("/scenarios/economics", response_model=ResultOut, tags=["scenarios"],
             summary="Lending economics, break-even PD and rate, sensitivity, PD × rate heatmap")
def economics(body: EconomicsRequest, _: Principal = Depends(analyst)) -> dict[str, Any]:
    a = fc.LendingAssumptions(**body.model_dump(exclude={"sensitivity_shock"}))
    try:
        a.validate()
        contrib = fc.lending_contribution(a)
        sens = fc.sensitivity_analysis(a, body.sensitivity_shock)
    except ValueError as exc:  # e.g. a shocked rate leaves its valid range
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(exc)) from exc
    pds = np.round(np.linspace(0.01, 0.30, 12), 4)
    rates = np.round(np.linspace(0.06, 0.36, 11), 4)
    heat = [{"pd": float(p), "annual_rate": float(r),
             "contribution_margin_on_exposure": fc.lending_contribution(replace(a, pd=float(p), annual_rate=float(r)))[
                 "contribution_margin_on_exposure"]} for p in pds for r in rates]
    pricing = [{"pd": float(p), "break_even_rate": fc.break_even_rate(replace(a, pd=float(p)))}
               for p in (0.01, 0.025, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30)]
    sim = [P.SIMULATED, P.ASSUMPTION]
    prov = {k: sim for k in ("total_exposure", "pd_over_horizon", "expected_interest_income", "fee_income",
                             "funding_cost", "expected_credit_loss", "operating_cost", "expected_contribution",
                             "contribution_per_loan", "contribution_margin_on_exposure", "el_rate", "break_even_pd",
                             "break_even_rate", "change_if_driver_down", "change_if_driver_up", "range")}
    prov |= {"pd": [P.ASSUMPTION], "annual_rate": [P.ASSUMPTION]}
    return {"result": sanitize({"contribution": contrib, "sensitivity": _records(sens), "heatmap": heat,
                                "pricing_floor": pricing, "assumptions": a.to_dict()}),
            "data_provenance": "observed", "provenance": prov}


@router.post("/scenarios/stress", response_model=ResultOut, tags=["scenarios"],
             summary="Stress test the held-out portfolio")
def stress(body: StressRequest, _: Principal = Depends(analyst), model: LoadedModel = Depends(deps.model),
           policy: Policy = Depends(deps.policy)) -> dict[str, Any]:
    port, a = _portfolio(model, policy, body.split, body.lgd, body.ccf)
    eco = policy.economics
    scenarios = [s.model_dump() for s in body.scenarios] if body.scenarios else fc.DEFAULT_SCENARIOS
    try:
        table = fc.stress_test(port["pd"].to_numpy(), a["lgd"], port["ead"].to_numpy(), scenarios,
                               body.annual_rate if body.annual_rate is not None else eco.annual_rate,
                               body.funding_rate if body.funding_rate is not None else eco.funding_rate,
                               body.opex_per_loan if body.opex_per_loan is not None else eco.opex_per_loan,
                               body.horizon_years if body.horizon_years is not None else eco.horizon_years)
    except ValueError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(exc)) from exc
    sim = labels(model, P.SIMULATED, P.ASSUMPTION)
    prov = {k: sim for k in table.columns if k != "scenario"}
    prov |= {k: [P.ASSUMPTION] for k in ("pd_multiplier", "lgd_shift", "ead_multiplier", "lgd", "ccf")}
    prov["n_loans"] = labels(model, P.OBSERVED)
    return {"result": {"split": body.split, "assumptions": a, "scenarios": _records(table)},
            "data_provenance": data_provenance(model), "provenance": prov}


@router.post("/scenarios/npv", response_model=ResultOut, tags=["scenarios"], summary="Loan NPV with cash flows")
def npv(body: NPVRequest, _: Principal = Depends(analyst)) -> dict[str, Any]:
    try:
        r = fc.loan_npv(**body.model_dump())
    except ValueError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(exc)) from exc
    flows = _records(r.pop("cash_flows"))
    sim = [P.SIMULATED, P.ASSUMPTION]
    prov = {k: sim for k in ("monthly_payment", "monthly_hazard", "lifetime_pd", "npv_contractual",
                             "npv_risk_adjusted", "pv_credit_cost", "expected_annual_yield",
                             "contractual_annual_yield", "contractual_cash_flow", "expected_payments",
                             "expected_recoveries", "expected_servicing_cost", "expected_cash_flow",
                             "discount_factor", "pv_expected")}
    prov["month"] = [P.ASSUMPTION]
    return {"result": sanitize({**r, "cash_flows": flows}), "data_provenance": "observed", "provenance": prov}


# ---------------------------------------------------------------------------- fairness

@router.get("/fairness/report", response_model=ResultOut, tags=["fairness"],
            summary="Group metrics, disparities, proxy scan, missingness by group")
def fairness_report(attribute: str | None = Query(None, description="Sensitive attribute (default: the first)"),
                    split: str = Split, _: Principal = Depends(analyst), model: LoadedModel = Depends(deps.model),
                    policy: Policy = Depends(deps.policy)) -> dict[str, Any]:
    """Audit only: the attribute is not a model input. Uses the active policy's PD thresholds. No single metric can
    establish fairness, and criteria can conflict when base rates differ between groups."""
    snap = model.snapshot
    attrs = list(snap.sensitive_cols)
    if not attrs:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "This model version kept no sensitive attributes for audit.")
    attribute = attribute or attrs[0]
    if attribute not in attrs:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, f"Choose one of: {', '.join(attrs)}")
    frame = snap.split(split)
    from creditiq_core.policy.engine import thresholds

    thr, _src = thresholds(policy, frame, model.artifact.threshold)
    groups = fair.make_groups(frame[attribute], attribute)
    # group_metrics takes one threshold; apply per-row thresholds by flagging first
    flagged = frame["pd"].to_numpy() >= thr
    table = fair.group_metrics(frame["y"], np.where(flagged, 1.0, 0.0), 0.5, groups)
    pd_table = fair.group_metrics(frame["y"], frame["pd"], 0.5, groups)  # for PD-based calibration and AUC
    for col in ("mean_pd", "calibration_gap", "roc_auc"):
        table[col] = table["group"].map(pd_table.set_index("group")[col])
    disparity = fair.disparity_summary(table)
    preds = [c for c in snap.predictors if c in frame.columns]
    proxy = fair.proxy_scan(frame, groups, preds)
    missing = fair.missingness_by_group(frame, groups, preds)
    obs, meas = labels(model, P.OBSERVED), labels(model, P.MEASURED)
    prov = {k: meas for k in ("flag_rate", "within_threshold_rate", "tpr_recall", "fnr_missed_defaults",
                              "fpr_good_flagged", "precision", "roc_auc", "calibration_gap", "highest", "lowest",
                              "difference", "ratio_low_to_high", "association", "max_gap")}
    prov |= {"n": obs, "share": obs, "observed_default_rate": obs, "mean_pd": labels(model, P.MODEL)}
    prov |= {str(g): obs for g in missing.columns if g not in ("feature", "max_gap")}
    return {"result": {"attribute": attribute, "available_attributes": attrs, "split": split,
                       "policy_version": policy.version, "groups": _records(table),
                       "disparities": _records(disparity), "proxy_scan": _records(proxy),
                       "missingness": _records(missing), "min_group_size": fair.MIN_GROUP_SIZE,
                       "guidance": FAIRNESS_GUIDANCE},
            "data_provenance": data_provenance(model), "provenance": prov}


FAIRNESS_GUIDANCE = [
    "Sensitive attributes are excluded from the model and used here only to audit outcomes.",
    "Other inputs can act as proxies; the proxy scan measures how strongly each input is associated with the groups.",
    "Groups with fewer than 30 applicants or 10 defaults are marked too small; their rates are noisy.",
    "Meeting one fairness criterion can violate another when base rates differ; there is no single 'fair' number.",
    "A disparity is a reason for human review of the policy and data, not an automatic conclusion.",
    "This audit is not legal advice and does not establish compliance with any regulation.",
]
