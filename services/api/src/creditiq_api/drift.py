"""Drift checks against the served model's training reference; results go to Postgres.

Used by the scheduled monitor job (``creditiq-monitor``) and by ``POST /v1/monitoring/check``.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import select

from creditiq_api.db import Alert, Database, MonitoringRun
from creditiq_api.model_store import LoadedModel
from creditiq_core import monitoring as mon
from creditiq_core.data.validation import apply_target_mapping
from creditiq_core.evaluation import classification_metrics
from creditiq_core.jsonsafe import sanitize

log = logging.getLogger(__name__)


def _labels(model: LoadedModel, batch: pd.DataFrame) -> np.ndarray | None:
    target = model.metadata.get("target", {})
    col = target.get("column")
    if not col or col not in batch.columns:
        return None
    pos = target.get("positive_values (default)", [])
    neg = target.get("negative_values (non-default)", [])
    y = apply_target_mapping(batch[col], pos, neg)
    if y.notna().sum() < 50 or y.dropna().nunique() < 2:
        return None
    return y.to_numpy()


def run_check(model: LoadedModel, batch: pd.DataFrame, batch_name: str, source: str, db: Database,
              thresholds: mon.DriftThresholds | None = None) -> dict[str, Any]:
    t = thresholds or mon.DriftThresholds()
    ref = model.artifact.reference
    schema = mon.schema_check(model.input_columns, batch, ref)
    features = mon.feature_drift(ref, batch, t)
    pd_new = model.artifact.predict_pd(batch.reindex(columns=model.input_columns))
    score = mon.score_drift(ref, pd_new, t)
    alerts = mon.drift_alert_records(features, score, schema, t)
    y = _labels(model, batch)
    perf: dict[str, Any] = {"labels_available": y is not None, "observed_default_rate": None, "roc_auc": None}
    if y is not None:
        mask = ~np.isnan(y)
        m = classification_metrics(y[mask].astype(int), pd_new[mask], model.artifact.threshold)
        perf |= {"observed_default_rate": m["prevalence"], "roc_auc": m["roc_auc"],
                 "calibration_in_the_large": float(pd_new[mask].mean() - m["prevalence"])}
        test_auc = model.test_metrics.get("roc_auc")
        if test_auc is not None and m["roc_auc"] is not None and test_auc - m["roc_auc"] >= 0.05:
            alerts.append(mon.DriftAlert("performance_drop", "critical",
                                         f"ROC-AUC on the labelled batch is {m['roc_auc']:.3f}, "
                                         f"{test_auc - m['roc_auc']:.3f} below the held-out test value.",
                                         float(m["roc_auc"]), float(test_auc) - 0.05))
    status = "critical" if any(a.severity == "critical" for a in alerts) else ("warning" if alerts else "ok")
    feat_rows = sanitize(features.astype(object).where(features.notna(), None).to_dict(orient="records"))
    psi = features["psi"].dropna() if not features.empty else pd.Series(dtype=float)
    with db.session() as s:
        run = MonitoringRun(source=source, model_name=model.name, model_version=model.version,
                            batch_name=batch_name[:200], n_rows=len(batch), status=status,
                            max_feature_psi=float(psi.max()) if len(psi) else None,
                            n_significant=int((psi >= t.psi_significant).sum()),
                            score_psi=float(score["psi"]) if score else None,
                            mean_pd=float(np.mean(pd_new)), ref_mean_pd=score["ref_mean_pd"] if score else None,
                            labels_available=perf["labels_available"],
                            observed_default_rate=perf["observed_default_rate"], roc_auc=perf["roc_auc"],
                            features=feat_rows, score=sanitize(score), schema_check=schema,
                            thresholds={"psi_moderate": t.psi_moderate, "psi_significant": t.psi_significant,
                                        "score_psi": t.score_psi, "missing_change": t.missing_change})
        s.add(run)
        s.flush()
        for a in alerts:
            s.add(Alert(run_id=run.id, code=a.code, severity=a.severity, message=a.message, value=a.value,
                        threshold=a.threshold, columns=list(a.columns)))
        run_id = run.id
    if alerts:
        log.warning("drift alerts", extra={"run_id": run_id, "status": status,
                                           "alerts": [a.code for a in alerts], "model_version": model.version})
    return run_to_dict(fetch_run(db, run_id))


def fetch_run(db: Database, run_id: int | None = None) -> tuple[MonitoringRun, list[Alert]] | None:
    with db.session() as s:
        q = select(MonitoringRun)
        q = q.where(MonitoringRun.id == run_id) if run_id is not None else q.order_by(MonitoringRun.id.desc())
        run = s.scalars(q.limit(1)).first()
        if run is None:
            return None
        alerts = list(s.scalars(select(Alert).where(Alert.run_id == run.id).order_by(Alert.id)))
        return run, alerts


def run_to_dict(found: tuple[MonitoringRun, list[Alert]] | None, with_detail: bool = True) -> dict[str, Any]:
    if found is None:
        return {}
    run, alerts = found
    d: dict[str, Any] = {c: getattr(run, c) for c in (
        "id", "created_at", "source", "model_name", "model_version", "batch_name", "n_rows", "status",
        "max_feature_psi", "n_significant", "score_psi", "mean_pd", "ref_mean_pd", "labels_available",
        "observed_default_rate", "roc_auc")}
    d["alerts"] = [{"id": a.id, "code": a.code, "severity": a.severity, "message": a.message, "value": a.value,
                    "threshold": a.threshold, "columns": a.columns, "created_at": a.created_at} for a in alerts]
    if with_detail:
        d |= {"features": run.features, "score": run.score, "schema_check": run.schema_check,
              "thresholds": run.thresholds}
    return d
