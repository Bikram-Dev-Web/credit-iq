"""Model card generated from a training run's real metadata (no placeholder metrics)."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from creditiq_core._frames import iter_rows


def fmt(v: Any) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, float | np.floating):
        if not np.isfinite(v):
            return "n/a"
        if abs(float(v)) >= 1000:
            return f"{v:,.0f}"
        return f"{v:.4g}"
    if isinstance(v, bool | np.bool_):
        return "yes" if v else "no"
    return str(v)


def pct(v: Any) -> str:
    return "n/a" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:.2%}"


def table_md(df: pd.DataFrame, max_rows: int = 40) -> str:
    df = df.head(max_rows)
    cols = [str(c) for c in df.columns]
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for row in iter_rows(df, index=False):
        out.append("| " + " | ".join(fmt(v).replace("|", "/") for v in row) + " |")
    return "\n".join(out)



def model_card_markdown(meta: dict[str, Any], comparison: pd.DataFrame | None = None) -> str:
    """Model card generated from the actual experiment metadata (no placeholder metrics)."""
    key = meta.get("artifact_model") or meta.get("bundle_model") or meta.get("selected_model")
    model = meta.get("models", {}).get(key, {})
    test = meta.get("metrics", {}).get(key, {}).get("test", {})
    ds = meta.get("dataset", {})
    tgt = meta.get("target", {})
    split = meta.get("split", {})
    cfg = meta.get("config", {})
    lines = [f"# Model card — {model.get('label', key)}", "",
             f"_Experiment {meta.get('experiment_id')} · created {meta.get('created_at')}_", "",
             "## Intended use",
             "Educational decision support: estimating the probability of the dataset's defined default event, explaining it, "
             "and illustrating expected loss and lending trade-offs.", "",
             "## Out-of-scope use",
             "- Automated approval, decline or pricing of real applicants.",
             "- Regulatory capital, IFRS 9 / CECL provisioning or any compliance determination.",
             "- Use on populations, products or periods different from the training data without re-validation.", "",
             "## Data",
             f"- Dataset: {ds.get('name')} ({'SYNTHETIC — fabricated records' if ds.get('is_synthetic') else ds.get('source')}), "
             f"{fmt(ds.get('n_rows_loaded'))} rows loaded.",
             f"- Target: `{tgt.get('column')}` with default label(s) {tgt.get('positive_values (default)')}; "
             f"prevalence {pct(tgt.get('prevalence'))}.",
             f"- Definition: {tgt.get('definition') or 'not documented'}",
             f"- Predictors: {len(meta.get('predictors', []))}; excluded columns: {len(meta.get('excluded_columns', {}))} "
             f"(identifiers, leakage-suspect, dates, sensitive attributes and user exclusions).",
             f"- Sensitive attributes kept only for auditing: {', '.join(meta.get('sensitive_columns_for_audit', [])) or 'none'}.", "",
             "## Model design",
             f"- Algorithm: {model.get('label')} (backend: {model.get('backend')}); parameters: `{model.get('params')}`.",
             f"- Preprocessing (fitted on training rows only): schema enforcement, domain ratio features "
             f"({', '.join(model.get('derived_features', [])) or 'none'}), median imputation with missing indicators, "
             "one-hot encoding with rare-level grouping" + (", standardisation" if key == "logreg" else "") + ".",
             f"- Class imbalance: class_weight = {cfg.get('class_weight')}.",
             f"- Calibration: {model.get('calibration')}.",
             f"- Decision threshold: {pct(model.get('threshold'))} ({cfg.get('threshold_rule')} on validation).", "",
             "## Validation (held-out test split)",
             f"- Split: {split.get('strategy')}; sizes {split.get('sizes')}.",
             f"- ROC-AUC {fmt(test.get('roc_auc'))}, Gini {fmt(test.get('gini'))}, PR-AUC {fmt(test.get('pr_auc'))} "
             f"(prevalence {pct(test.get('prevalence'))}), Brier {fmt(test.get('brier'))}, KS {fmt(test.get('ks'))}.",
             f"- At the threshold: recall {pct(test.get('recall'))}, precision {pct(test.get('precision'))}, "
             f"specificity {pct(test.get('specificity'))}, flagged {pct(test.get('flag_rate'))} of applicants.",
             ]
    if ds.get("is_synthetic"):
        lines.append("- These figures come from synthetic data and are not evidence of real-world performance.")
    if comparison is not None and len(comparison):
        lines += ["", "### Comparison with other trained models", "",
                  table_md(comparison[["model", "val_roc_auc", "test_roc_auc", "test_pr_auc", "test_brier"]])]
    lines += ["", "## Limitations",
              "- PD refers to the dataset's own target event and window; mapping it to a 12-month PD is an assumption.",
              "- Explanations are associational, not causal; attributions are unstable for correlated inputs.",
              "- Random splits do not test performance on future cohorts.", "",
              "## Fairness considerations",
              "Sensitive attributes are excluded from inputs by default but remaining inputs can act as proxies. Group-level "
              "error rates and calibration should be reviewed on the Fairness page; no single metric "
              "establishes fairness, and legal requirements vary by jurisdiction.", "",
              "## Governance requirements before any real use",
              "- Independent validation including out-of-time testing and benchmarking.",
              "- Legal review of inputs and adverse-action reasoning.",
              "- Human review of flagged and declined applications; documented override process.",
              "- Ongoing monitoring of drift, calibration and group outcomes with defined escalation thresholds."]
    return "\n".join(lines) + "\n"
