"""Drift monitoring: latest run, history, alerts and on-demand checks."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query, Request, UploadFile, status
from sqlalchemy import select

from creditiq_api import deps, drift
from creditiq_api.db import Alert, Database, MonitoringRun
from creditiq_api.model_store import LoadedModel
from creditiq_api.schemas import COMMON_ERRORS, ResultOut
from creditiq_api.security import Principal, analyst
from creditiq_core.data.csv import read_csv_file
from creditiq_core.provenance import Provenance as P

router = APIRouter(prefix="/v1/monitoring", tags=["monitoring"], responses=COMMON_ERRORS)

DRIFT_KEYS = ("psi", "max_feature_psi", "score_psi", "ref_missing_rate", "new_missing_rate", "missing_change",
              "n_significant", "value", "threshold", "psi_moderate", "psi_significant", "missing_change")


def _prov(model: LoadedModel | None, synthetic: bool) -> dict[str, list[P]]:
    extra = [P.SYNTHETIC] if synthetic else []
    meas = [P.MEASURED, *extra]
    prov = {k: meas for k in DRIFT_KEYS}
    prov |= {"n_rows": [P.OBSERVED, *extra], "observed_default_rate": [P.OBSERVED, *extra],
             "roc_auc": meas, "mean_pd": [P.MODEL, *extra], "ref_mean_pd": [P.MODEL, *extra],
             "new_mean_pd": [P.MODEL, *extra], "ref_props": meas, "new_props": meas, "cuts": meas,
             "n_input_columns": [P.OBSERVED], "n_present": [P.OBSERVED], "score_psi_threshold": [P.ASSUMPTION],
             "calibration_in_the_large": meas, "id": [P.OBSERVED], "warning": [P.OBSERVED],
             "critical": [P.OBSERVED], "ok": [P.OBSERVED]}
    return prov


def _out(result: Any, model: LoadedModel | None) -> dict[str, Any]:
    synthetic = model.is_synthetic if model else True
    return {"result": result, "data_provenance": "synthetic" if synthetic else "observed",
            "provenance": _prov(model, synthetic)}


@router.get("/drift", response_model=ResultOut, summary="Latest drift check (PSI per feature and for PD)")
def latest(request: Request, _: Principal = Depends(analyst), db: Database = Depends(deps.db)) -> dict[str, Any]:
    return _out(drift.run_to_dict(drift.fetch_run(db)) or None, request.app.state.holder.current)


@router.get("/history", response_model=ResultOut, summary="Logged drift checks, newest first")
def history(request: Request, limit: int = Query(50, ge=1, le=500), _: Principal = Depends(analyst),
            db: Database = Depends(deps.db)) -> dict[str, Any]:
    with db.session() as s:
        runs = list(s.scalars(select(MonitoringRun).order_by(MonitoringRun.id.desc()).limit(limit)))
        ids = [r.id for r in runs]
        alerts = list(s.scalars(select(Alert).where(Alert.run_id.in_(ids)))) if ids else []
    by_run: dict[int, list[Alert]] = {}
    for a in alerts:
        by_run.setdefault(a.run_id, []).append(a)
    rows = [drift.run_to_dict((r, by_run.get(r.id, [])), with_detail=False) for r in runs]
    return _out(rows, request.app.state.holder.current)


@router.get("/alerts", response_model=ResultOut, summary="Alert history, newest first")
def alerts(request: Request, limit: int = Query(100, ge=1, le=1000), _: Principal = Depends(analyst),
           db: Database = Depends(deps.db)) -> dict[str, Any]:
    with db.session() as s:
        rows = s.execute(select(Alert, MonitoringRun.model_version, MonitoringRun.batch_name)
                         .join(MonitoringRun, Alert.run_id == MonitoringRun.id)
                         .order_by(Alert.id.desc()).limit(limit)).all()
    out = [{"id": a.id, "run_id": a.run_id, "created_at": a.created_at, "code": a.code, "severity": a.severity,
            "message": a.message, "value": a.value, "threshold": a.threshold, "columns": a.columns,
            "model_version": mv, "batch_name": bn} for a, mv, bn in rows]
    return _out(out, request.app.state.holder.current)


@router.post("/check", response_model=ResultOut, summary="Run a drift check on an uploaded CSV batch")
async def check(request: Request, file: UploadFile, _: Principal = Depends(analyst),
                model: LoadedModel = Depends(deps.model), db: Database = Depends(deps.db)) -> dict[str, Any]:
    """Compares the batch with the served model's training reference (features) and validation PDs (score). If the
    batch has the training target column, performance is re-measured too. The run and its alerts are logged."""
    data = await file.read(request.app.state.settings.batch_max_bytes * 4 + 1)
    if len(data) > request.app.state.settings.batch_max_bytes * 4:
        raise HTTPException(status.HTTP_413_CONTENT_TOO_LARGE, "Monitoring batch is too large.")
    try:
        frame, _info = read_csv_file(data, name=file.filename or "batch.csv", optimize_memory=False)
    except ValueError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, f"Unreadable CSV: {exc}") from exc
    result = drift.run_check(model, frame, file.filename or "upload.csv", "api", db)
    return _out(result, model)



@router.post("/replay-demo", response_model=ResultOut, summary="Run a drift check on the demo batch for the served model")
def replay_demo(request: Request, _: Principal = Depends(analyst), model: LoadedModel = Depends(deps.model),
                db: Database = Depends(deps.db)) -> dict[str, Any]:
    """Uses the first configured demo batch (`DEMO_BATCHES`) whose columns cover the served model's inputs, so the
    demo works for the synthetic model and for a model trained on real data alike."""
    needed = set(model.input_columns)
    for raw in request.app.state.settings.demo_batches.split(","):
        path = Path(raw.strip())
        if not path.is_file():
            continue
        header = set(pd.read_csv(path, nrows=0).columns)
        if needed <= header:
            frame, _info = read_csv_file(path.read_bytes(), name=path.name, optimize_memory=False)
            return _out(drift.run_check(model, frame, path.name, "replay", db), model)
    raise HTTPException(status.HTTP_404_NOT_FOUND,
                        "No demo batch matches the served model's inputs. Upload a CSV with 'Check a CSV batch', or "
                        "list a matching file in DEMO_BATCHES.")
