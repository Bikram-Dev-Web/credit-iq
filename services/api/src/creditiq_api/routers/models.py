"""GET /v1/models…, POST /v1/models/{version}/promote."""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, status

from creditiq_api import deps
from creditiq_api.db import AdminEvent, Database
from creditiq_api.model_store import LoadedModel, ModelHolder, load_package
from creditiq_api.schemas import COMMON_ERRORS, ModelList, ModelSchemaOut, PromoteRequest, PromoteResponse, ResultOut
from creditiq_api.scoring import data_provenance, labels
from creditiq_api.security import Principal, admin, analyst
from creditiq_core.provenance import Provenance as P

router = APIRouter(prefix="/v1/models", tags=["models"], responses=COMMON_ERRORS)

VERSION_PROV = [P.MEASURED]


def _version_model(request: Request, holder: ModelHolder, version: str) -> LoadedModel:
    """The served model if it is that version; otherwise load the version read-only (cached per process)."""
    cur = holder.current
    if cur is not None and cur.version == version:
        return cur
    cache: dict[str, LoadedModel] = request.app.state.version_cache
    if version not in cache:
        try:
            path = holder.registry.fetch(version)
        except (KeyError, FileNotFoundError) as exc:
            raise HTTPException(status.HTTP_404_NOT_FOUND, f"Model version {version} not found.") from exc
        except Exception as exc:  # MLflow raises its own exception types
            raise HTTPException(status.HTTP_404_NOT_FOUND, f"Model version {version} not found: {exc}") from exc
        if len(cache) >= 4:
            cache.pop(next(iter(cache)))
        cache[version] = load_package(holder.registry.model_name, version, path)
    return cache[version]


@router.get("", response_model=ModelList, summary="Registered versions, aliases and headline metrics")
def list_models(request: Request, _: Principal = Depends(analyst),
                holder: ModelHolder = Depends(deps.holder)) -> dict[str, Any]:
    cur = holder.current
    versions = []
    for v in holder.registry.versions():
        d = v.to_dict()
        d["serving"] = cur is not None and cur.version == v.version
        versions.append(d)
    synthetic = all(v.get("is_synthetic", True) for v in versions) if versions else True
    return {"model_name": holder.registry.model_name, "serving_alias": holder.alias,
            "serving_version": cur.version if cur else None, "versions": versions,
            "data_provenance": "synthetic" if synthetic else "observed",
            "provenance": {k: [P.MEASURED] + ([P.SYNTHETIC] if synthetic else [])
                           for k in ("test_roc_auc", "test_pr_auc", "test_brier")} | {"created": [P.OBSERVED]}}


@router.get("/{version}", response_model=ResultOut, summary="One version: metadata, metrics, lineage")
def get_model(version: str, request: Request, _: Principal = Depends(analyst),
              holder: ModelHolder = Depends(deps.holder)) -> dict[str, Any]:
    m = _version_model(request, holder, version)
    ev = m.evaluation
    meta = m.metadata
    result = {
        "model_name": m.name, "version": m.version, "selected_model": m.key, "label": m.label,
        "serving": holder.current is not None and holder.current.version == m.version,
        "is_calibrated": m.artifact.model.is_calibrated, "threshold": m.artifact.threshold,
        "threshold_rule": m.artifact.threshold_rule, "threshold_source": m.artifact.threshold_source,
        "input_columns": m.input_columns, "dataset": meta.get("dataset"), "target": meta.get("target"),
        "split": meta.get("split"), "config": meta.get("config"), "lineage": meta.get("lineage", {}),
        "excluded_columns": meta.get("excluded_columns"),
        "sensitive_columns_for_audit": meta.get("sensitive_columns_for_audit"),
        "comparison": ev.get("comparison"), "warnings": ev.get("warnings"),
        "leakage_guard_excluded": ev.get("leakage_guard_excluded"), "split_sizes": ev.get("split_sizes"),
        "test_metrics": m.test_metrics,
    }
    measured = labels(m, P.MEASURED)
    prov = {k: measured for k in ("roc_auc", "gini", "pr_auc", "brier", "log_loss", "ks", "precision", "recall",
                                  "specificity", "f1", "balanced_accuracy", "accuracy", "flag_rate", "mean_pd",
                                  "lift_top10", "val_roc_auc", "val_pr_auc", "val_brier", "test_roc_auc",
                                  "test_gini", "test_pr_auc", "test_brier", "test_log_loss", "test_ks",
                                  "test_recall", "test_precision", "test_specificity", "test_f1",
                                  "test_balanced_accuracy", "train_roc_auc", "overfit_gap_auc", "flagged", "n",
                                  "tp", "fp", "tn", "fn")}
    prov |= {"prevalence": labels(m, P.OBSERVED), "threshold": [P.MEASURED], "fit_seconds": [P.MEASURED],
             "n_rows_loaded": labels(m, P.OBSERVED), "n_cols": labels(m, P.OBSERVED),
             "n_rows_file": labels(m, P.OBSERVED), "n_positive": labels(m, P.OBSERVED),
             "n_negative": labels(m, P.OBSERVED), "n_unmapped_dropped": labels(m, P.OBSERVED),
             "train": labels(m, P.OBSERVED), "validation": labels(m, P.OBSERVED), "test": labels(m, P.OBSERVED)}
    return {"result": result, "data_provenance": data_provenance(m), "provenance": prov}


@router.get("/{version}/schema", response_model=ModelSchemaOut, summary="Input schema for the scoring form")
def get_schema(version: str, request: Request, _: Principal = Depends(analyst),
               holder: ModelHolder = Depends(deps.holder)) -> dict[str, Any]:
    """Fields in order of influence (the model's internal importance; engineered ratios also credit their source
    inputs). The 10 most influential fields form the main scoring form; the rest are 'advanced'."""
    m = _version_model(request, holder, version)
    importance = m.importance
    order = sorted(m.input_columns, key=lambda c: -importance.get(c, 0.0))
    fields = []
    for rank, col in enumerate(order, start=1):
        info = m.artifact.input_schema[col]
        f: dict[str, Any] = {"name": col, "kind": info["kind"], "importance_rank": rank,
                             "importance": importance.get(col), "missing_rate": info.get("missing_rate", 0.0),
                             "advanced": rank > 10}
        if info["kind"] == "numeric":
            f |= {k: info.get(k) for k in ("min", "max", "p01", "p99", "median", "is_integer")}
        else:
            f |= {"categories": info.get("categories", []), "mode": info.get("mode")}
        fields.append(f)
    return {"model_version": m.version, "exposure_field": m.exposure_col, "fields": fields,
            "json_schema": m.request_model.model_json_schema(), "data_provenance": data_provenance(m),
            "provenance": {k: labels(m, P.OBSERVED) for k in ("min", "max", "p01", "p99", "median", "missing_rate")}
            | {"importance": labels(m, P.MEASURED), "importance_rank": labels(m, P.MEASURED)}}


@router.get("/{version}/evaluation", response_model=ResultOut, summary="Curves, calibration, deciles, threshold sweep")
def get_evaluation(version: str, request: Request, _: Principal = Depends(analyst),
                   holder: ModelHolder = Depends(deps.holder)) -> dict[str, Any]:
    """ROC, PR, calibration (with Wilson intervals) and deciles per model and split, and the threshold sweep on
    the VALIDATION split only (thresholds are chosen there; test results are reported, never tuned on)."""
    m = _version_model(request, holder, version)
    measured = labels(m, P.MEASURED)
    keys = ("fpr", "tpr", "threshold", "recall", "precision", "pd_from", "pd_to", "mean_pd", "observed_rate",
            "ci_low", "ci_high", "ece", "min_pd", "max_pd", "default_rate", "lift", "cum_capture", "cum_lift",
            "flag_rate", "specificity", "f1", "balanced_accuracy", "roc_auc", "gini", "pr_auc", "brier",
            "log_loss", "ks", "accuracy", "lift_top10", "fit_seconds", "val_roc_auc", "val_pr_auc", "val_brier",
            "test_roc_auc", "test_gini", "test_pr_auc", "test_brier", "test_log_loss", "test_ks", "test_recall",
            "test_precision", "test_specificity", "test_f1", "test_balanced_accuracy", "train_roc_auc",
            "overfit_gap_auc", "n", "defaults", "bin", "risk_group", "flagged", "tp", "fp", "tn", "fn",
            "train", "validation", "test")
    prov = {k: measured for k in keys} | {"prevalence": labels(m, P.OBSERVED)}
    return {"result": m.evaluation, "data_provenance": data_provenance(m), "provenance": prov}


@router.get("/{version}/card", response_model=ResultOut, summary="Model card (Markdown in result.markdown)")
def get_card(version: str, request: Request, _: Principal = Depends(analyst),
             holder: ModelHolder = Depends(deps.holder)) -> dict[str, Any]:
    m = _version_model(request, holder, version)
    return {"result": {"version": m.version, "markdown": m.model_card}, "data_provenance": data_provenance(m),
            "provenance": {}}


@router.post("/{version}/promote", response_model=PromoteResponse, summary="Move an alias to this version (admin)")
def promote(version: str, request: Request, body: PromoteRequest | None = None, principal: Principal = Depends(admin),
            holder: ModelHolder = Depends(deps.holder), db: Database = Depends(deps.db)) -> dict[str, Any]:
    """Promoting to `production` keeps the old production version as `previous` (one-step rollback) and hot-reloads
    the served model: the new version is loaded and warmed, then swapped in; requests in flight are unaffected."""
    alias = (body or PromoteRequest()).alias
    try:
        _version_model(request, holder, version)  # validates the package before any alias moves
        moved = holder.registry.set_alias(alias, version)
    except KeyError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Model version {version} not found.") from exc
    with db.session() as s:
        s.add(AdminEvent(actor=principal.key_id, action="promote_model",
                         detail={"model": holder.registry.model_name, **moved}))
    if alias == holder.alias:
        holder.load_version(version, "promotion")
    cur = holder.current
    return {"model_name": holder.registry.model_name, **moved, "serving_version": cur.version if cur else None}
