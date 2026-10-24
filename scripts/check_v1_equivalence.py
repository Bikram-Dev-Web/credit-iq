"""Check that creditiq_core reproduces v1 (legacy/src) exactly on the same data and seed.

Usage:  uv run python scripts/check_v1_equivalence.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "legacy"))  # v1 code, imported as `src`
from src import data_validation as v1dv
from src import explainability as v1x
from src import financial_calculations as v1fc
from src import modeling as v1m

from creditiq_core import explainability as x
from creditiq_core.data import synthetic
from creditiq_core.data import validation as dv
from creditiq_core.finance import economics as fc
from creditiq_core.models import training as m

df = synthetic.generate_synthetic_credit_data(6000, seed=21)
roles = dv.suggest_feature_roles(df, "TARGET", "APPLICATION_DATE")
preds = roles.loc[roles["use"], "column"].tolist()
assert preds == v1dv.suggest_feature_roles(df, "TARGET", "APPLICATION_DATE").query("use")["column"].tolist()
kw = dict(date_col="APPLICATION_DATE", exposure_col="AMT_CREDIT", sensitive_cols=["CODE_GENDER"], excluded={"SK_ID_CURR": "identifier"})
r1 = v1m.run_experiment(df, {}, v1dv.build_target_mapping(df, "TARGET", [1], [0]), preds, v1m.TrainingConfig(sample_size=None, seed=5), **kw)
r2 = m.run_experiment(df, {}, dv.build_target_mapping(df, "TARGET", [1], [0]), preds, m.TrainingConfig(sample_size=None, seed=5), **kw)
assert r1.selected == r2.selected and r1.predictors == r2.predictors
worst = 0.0
for k in r1.models:
    for s in ("train", "validation", "test"):
        worst = max(worst, float(np.max(np.abs(r1.pds[k][s] - r2.pds[k][s]))))
    assert r1.thresholds[k] == r2.thresholds[k], k
    assert r1.metrics[k]["test"] == r2.metrics[k]["test"], k
row = r1.X_split("test").iloc[[3]]
for k in r1.models:
    e1, _ = v1x.local_explanation(r1.models[k], row, r1.schema, r1.bundle(k).transformed_mean)
    e2, _ = x.local_explanation(r2.models[k], row, r2.schema, r2.to_artifact(k).transformed_mean)
    assert np.allclose(e1["contribution"], e2["contribution"]) and list(e1["feature"]) == list(e2["feature"]), k
assert v1fc.loan_npv(100_000, 0.16, 24, 0.08, 0.45, 0.07)["npv_risk_adjusted"] == fc.loan_npv(100_000, 0.16, 24, 0.08, 0.45, 0.07)["npv_risk_adjusted"]
print(f"identical: selected={r2.selected}, thresholds equal, test metrics equal, max |PD diff|={worst:.1e}, explanations equal, NPV equal")
for k in r2.models:
    print(f"  {k:14s} test ROC-AUC {r2.metrics[k]['test']['roc_auc']:.4f}  threshold {r2.thresholds[k]:.4f}")
