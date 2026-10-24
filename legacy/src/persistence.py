"""Local persistence: SQLite metadata/audit log and joblib model bundles.

Everything is stored on the local disk in plain (unencrypted) files:

* ``storage/creditiq.db`` – experiment metadata, borrower-assessment audit log
  (scores only: no raw applicant inputs, just a one-way hash), monitoring runs.
* ``models/*.joblib`` – trained pipelines with calibration and summary statistics
  (no training rows).

joblib files are Python pickles: only load bundles that you created yourself.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from . import config
from .data_loader import sanitize_filename

SCHEMA = """
CREATE TABLE IF NOT EXISTS experiments (
    id TEXT PRIMARY KEY, created_at TEXT, dataset_name TEXT, is_synthetic INTEGER,
    dataset_fingerprint TEXT, target_column TEXT, positive_labels TEXT, n_rows INTEGER,
    predictors TEXT, excluded TEXT, split TEXT, config TEXT, selected_model TEXT,
    test_metrics TEXT, artifact_path TEXT
);
CREATE TABLE IF NOT EXISTS assessments (
    id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT, experiment_id TEXT, model TEXT,
    pd REAL, raw_score REAL, threshold REAL, risk_band TEXT, outcome TEXT, ead REAL,
    expected_loss REAL, input_hash TEXT
);
CREATE TABLE IF NOT EXISTS monitoring_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT, experiment_id TEXT, model TEXT,
    batch_name TEXT, n_rows INTEGER, max_feature_psi REAL, n_significant INTEGER, score_psi REAL,
    mean_pd REAL, observed_default_rate REAL, roc_auc REAL, alerts TEXT
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (pd.Timestamp, datetime)):
        return str(o)
    return str(o)


def to_json(obj, indent: int | None = None) -> str:
    """JSON with numpy support; NaN/inf become null so the output is always valid JSON."""
    return json.dumps(_sanitize(obj), default=_json_default, indent=indent)


def _sanitize(obj):
    if isinstance(obj, dict):
        return {str(k): _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    if isinstance(obj, np.floating):
        return None if not np.isfinite(obj) else float(obj)
    if isinstance(obj, np.ndarray):
        return _sanitize(obj.tolist())
    return obj


def connect(db_path: Path | None = None) -> sqlite3.Connection:
    db_path = db_path or config.DB_PATH
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.executescript(SCHEMA)
    return conn


def save_experiment(result, artifact_path: str | None = None, db_path: Path | None = None) -> None:
    test_metrics = {k: {m: v for m, v in result.metrics[k]["test"].items()
                        if m in ("roc_auc", "pr_auc", "brier", "recall", "precision", "threshold")}
                    for k in result.models}
    with closing(connect(db_path)) as conn, conn:
        conn.execute(
            "INSERT OR REPLACE INTO experiments VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (result.experiment_id, result.created_at, result.dataset.get("name"),
             int(bool(result.dataset.get("is_synthetic"))), result.dataset.get("fingerprint"),
             result.target.get("column"), to_json(result.target.get("positive_values (default)")),
             len(result.frame), to_json(result.predictors), to_json(result.excluded),
             to_json(result.split.to_dict()), to_json(result.config.to_dict()), result.selected,
             to_json(test_metrics), artifact_path),
        )


def update_experiment_artifact(experiment_id: str, artifact_path: str, db_path: Path | None = None) -> None:
    with closing(connect(db_path)) as conn, conn:
        conn.execute("UPDATE experiments SET artifact_path=? WHERE id=?", (artifact_path, experiment_id))


def list_experiments(db_path: Path | None = None) -> pd.DataFrame:
    with closing(connect(db_path)) as conn:
        df = pd.read_sql_query("SELECT * FROM experiments ORDER BY created_at DESC", conn)
    return df


def save_bundle(bundle, name: str, models_dir: Path | None = None) -> Path:
    models_dir = models_dir or config.MODELS_DIR
    models_dir.mkdir(parents=True, exist_ok=True)
    fname = sanitize_filename(name, "model")
    if not fname.endswith(".joblib"):
        fname += ".joblib"
    path = models_dir / fname
    joblib.dump(bundle, path, compress=3)
    return path


def bundle_bytes(bundle) -> bytes:
    import io
    buf = io.BytesIO()
    joblib.dump(bundle, buf, compress=3)
    return buf.getvalue()


def list_bundles(models_dir: Path | None = None) -> list[Path]:
    models_dir = models_dir or config.MODELS_DIR
    if not models_dir.exists():
        return []
    return sorted(models_dir.glob("*.joblib"), key=lambda p: p.stat().st_mtime, reverse=True)


def load_bundle(path: Path, models_dir: Path | None = None):
    """Load a bundle from the local models folder only (pickles can execute code)."""
    from .modeling import ModelBundle

    models_dir = Path(models_dir or config.MODELS_DIR).resolve()
    path = Path(path).resolve()
    if models_dir not in path.parents:
        raise ValueError("Model bundles can only be loaded from the local models/ folder.")
    obj = joblib.load(path)
    if not isinstance(obj, ModelBundle):
        raise ValueError("This file is not a CreditIQ model bundle.")
    return obj


def hash_inputs(inputs: dict) -> str:
    """One-way hash so an assessment can be matched later without storing applicant data."""
    return hashlib.sha256(to_json(inputs).encode()).hexdigest()[:16]


def log_assessment(record: dict, db_path: Path | None = None) -> None:
    with closing(connect(db_path)) as conn, conn:
        conn.execute(
            "INSERT INTO assessments (created_at, experiment_id, model, pd, raw_score, threshold, risk_band, "
            "outcome, ead, expected_loss, input_hash) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (_now(), record.get("experiment_id"), record.get("model"), record.get("pd"), record.get("raw_score"),
             record.get("threshold"), record.get("risk_band"), record.get("outcome"), record.get("ead"),
             record.get("expected_loss"), record.get("input_hash")),
        )


def list_assessments(limit: int = 200, db_path: Path | None = None) -> pd.DataFrame:
    with closing(connect(db_path)) as conn:
        return pd.read_sql_query("SELECT * FROM assessments ORDER BY id DESC LIMIT ?", conn, params=(limit,))


def log_monitoring_run(record: dict, db_path: Path | None = None) -> None:
    with closing(connect(db_path)) as conn, conn:
        conn.execute(
            "INSERT INTO monitoring_runs (created_at, experiment_id, model, batch_name, n_rows, max_feature_psi, "
            "n_significant, score_psi, mean_pd, observed_default_rate, roc_auc, alerts) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (_now(), record.get("experiment_id"), record.get("model"), record.get("batch_name"), record.get("n_rows"),
             record.get("max_feature_psi"), record.get("n_significant"), record.get("score_psi"), record.get("mean_pd"),
             record.get("observed_default_rate"), record.get("roc_auc"), to_json(record.get("alerts", []))),
        )


def list_monitoring_runs(experiment_id: str | None = None, db_path: Path | None = None) -> pd.DataFrame:
    with closing(connect(db_path)) as conn:
        if experiment_id:
            return pd.read_sql_query("SELECT * FROM monitoring_runs WHERE experiment_id=? ORDER BY id", conn, params=(experiment_id,))
        return pd.read_sql_query("SELECT * FROM monitoring_runs ORDER BY id", conn)


def clear_local_records(experiments: bool = False, assessments: bool = False, monitoring: bool = False,
                        models: bool = False, reports: bool = False, db_path: Path | None = None) -> dict:
    """Delete locally stored records/artifacts. Returns counts removed."""
    removed = {}
    with closing(connect(db_path)) as conn, conn:
        for flag, table in ((experiments, "experiments"), (assessments, "assessments"), (monitoring, "monitoring_runs")):
            if flag:
                removed[table] = conn.execute(f"DELETE FROM {table}").rowcount
    if models:
        files = list_bundles()
        for f in files:
            f.unlink(missing_ok=True)
        removed["model_files"] = len(files)
    if reports and config.REPORTS_DIR.exists():
        files = [p for p in config.REPORTS_DIR.iterdir() if p.is_file() and p.name != ".gitkeep"]
        for f in files:
            f.unlink(missing_ok=True)
        removed["report_files"] = len(files)
    return removed


def save_report(text: str, name: str, reports_dir: Path | None = None) -> Path:
    reports_dir = reports_dir or config.REPORTS_DIR
    reports_dir.mkdir(parents=True, exist_ok=True)
    path = reports_dir / sanitize_filename(name, "report.md")
    path.write_text(text, encoding="utf-8")
    return path


def storage_locations() -> dict:
    return {"Experiment & audit database": str(config.DB_PATH), "Model bundles": str(config.MODELS_DIR),
            "Saved reports": str(config.REPORTS_DIR), "Local data files": str(config.DATA_DIR)}

