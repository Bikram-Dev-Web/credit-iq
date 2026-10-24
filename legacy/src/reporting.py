"""Project report generator (Markdown and self-contained HTML).

Every section is labelled with what kind of evidence it contains, so a reader
can never confuse a measured model result with an assumption or a projection:

* MEASURED   – computed by evaluating the trained model on held-out data
* OBSERVED   – historical facts about the loaded dataset
* ASSUMPTION – values typed in by the user
* SIMULATED  – projections that combine model outputs with assumptions

Sections whose inputs do not exist yet are omitted; nothing is filled with
placeholder numbers.
"""
from __future__ import annotations

import html
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np
import pandas as pd

KINDS = {
    "MEASURED": "Measured model result (held-out data)",
    "OBSERVED": "Historical data observation",
    "ASSUMPTION": "User-defined assumption",
    "SIMULATED": "Simulated / projected outcome",
    "CONTEXT": "Methodology and context",
}


@dataclass
class ReportSection:
    title: str
    kind: str
    paragraphs: list[str] = field(default_factory=list)
    bullets: list[str] = field(default_factory=list)
    tables: list[tuple[str, pd.DataFrame]] = field(default_factory=list)


def _fmt(v) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, (float, np.floating)):
        if not np.isfinite(v):
            return "n/a"
        if abs(v) >= 1000:
            return f"{v:,.0f}"
        return f"{v:.4g}"
    if isinstance(v, (bool, np.bool_)):
        return "yes" if v else "no"
    return str(v)


def _pct(v) -> str:
    return "n/a" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:.2%}"


def build_report(ctx: dict) -> list[ReportSection]:
    """Assemble sections from whatever results currently exist in ``ctx``."""
    s: list[ReportSection] = []
    ds = ctx.get("dataset") or {}
    s.append(ReportSection(
        "Purpose and scope", "CONTEXT",
        paragraphs=["CreditIQ estimates the probability of a defined default/payment-difficulty event, "
                    "converts it into expected credit loss (PD × LGD × EAD), and compares lending policies "
                    "under explicit financial assumptions. It is an educational decision-support prototype, "
                    "not a production credit-decisioning or regulatory system."]))
    if ds:
        bullets = [f"Name: {ds.get('name')}", f"Source: {ds.get('source')}",
                   f"Rows loaded: {_fmt(ds.get('n_rows_loaded'))}; columns: {_fmt(ds.get('n_cols'))}",
                   f"Synthetic: {'YES — fabricated demonstration data, not real borrowers' if ds.get('is_synthetic') else 'no'}"]
        bullets += [f"Note: {n}" for n in ds.get("notes", [])]
        s.append(ReportSection("Dataset", "OBSERVED", bullets=bullets))
    tgt = ctx.get("target")
    if tgt:
        s.append(ReportSection("Target definition (user-confirmed mapping)", "OBSERVED", bullets=[
            f"Column: {tgt.get('column')}",
            f"Default label(s): {tgt.get('positive_values (default)')}",
            f"Non-default label(s): {tgt.get('negative_values (non-default)')}",
            f"Observed default prevalence: {_pct(tgt.get('prevalence'))} "
            f"({_fmt(tgt.get('n_positive'))} defaults / {_fmt(tgt.get('n_positive', 0) + tgt.get('n_negative', 0))} labelled)",
            f"Unmapped rows excluded: {_fmt(tgt.get('n_unmapped_dropped'))}",
            f"Definition: {tgt.get('definition') or 'not provided'}"]))
    q = ctx.get("quality")
    if q:
        s.append(ReportSection("Data quality", "OBSERVED", bullets=[f"{k}: {_fmt(v)}" for k, v in q.items()]))

    exp = ctx.get("experiment")
    if exp is not None:
        cfg = exp.config
        split = exp.split.to_dict()
        s.append(ReportSection("Methodology", "CONTEXT", bullets=[
            f"Split: {split['strategy']} — train {split['sizes']['train']:,}, validation {split['sizes']['validation']:,}, test {split['sizes']['test']:,}",
            *[f"Split note: {n}" for n in split["notes"]],
            f"Predictors used: {len(exp.predictors)}; excluded columns: {len(exp.excluded)}",
            f"Class imbalance handling: class_weight={cfg.class_weight}",
            f"Calibration: {cfg.calibration} (fitted on validation only)",
            f"Threshold rule: {cfg.threshold_rule} (chosen on validation only)",
            f"Model selection: best validation {cfg.selection_metric}",
            f"Hyperparameter tuning on validation: {'yes' if cfg.tune else 'no'}",
            "The test split was used only for the final metrics reported below."]))
        comp = exp.comparison_table()
        cols = ["model", "selected", "val_roc_auc", "test_roc_auc", "test_gini", "test_pr_auc", "test_brier",
                "test_ks", "threshold", "test_recall", "test_precision", "test_specificity", "overfit_gap_auc"]
        prev = exp.metrics[exp.selected]["test"]["prevalence"]
        s.append(ReportSection("Model comparison (test set)", "MEASURED",
                               paragraphs=[f"Selected model: {exp.selected_model.label}. Test-set default prevalence "
                                           f"{_pct(prev)} is the random baseline for PR-AUC."
                                           + (" These results come from SYNTHETIC data and are not evidence of real-world performance."
                                              if ds.get("is_synthetic") else "")],
                               tables=[("Held-out performance", comp[cols])]))
        if ctx.get("calibration_table") is not None:
            s.append(ReportSection("Calibration of the selected model (test set)", "MEASURED",
                                   paragraphs=[f"Expected calibration error: {_fmt(ctx.get('ece'))}."],
                                   tables=[("Predicted PD vs observed default rate by bin", ctx["calibration_table"])]))
    if ctx.get("explanations") is not None:
        s.append(ReportSection("Most influential features (selected model)", "MEASURED",
                               paragraphs=["Permutation importance: drop in validation ROC-AUC when the feature is shuffled. "
                                           "This reflects predictive association, not causation."],
                               tables=[("Top features", ctx["explanations"])]))
    fin = ctx.get("assumptions")
    if fin:
        s.append(ReportSection("Financial assumptions", "ASSUMPTION", bullets=[f"{k}: {_fmt(v)}" for k, v in fin.items()]))
    kpis = ctx.get("portfolio_kpis")
    if kpis:
        s.append(ReportSection("Portfolio expected loss", "SIMULATED", paragraphs=[
            "Computed on the out-of-sample test portfolio. Observed default rate is a historical fact; "
            "average PD is a model output; expected loss combines PD with the LGD/EAD assumptions."],
            bullets=[f"Loans: {_fmt(kpis.get('n_loans'))}", f"Total EAD: {_fmt(kpis.get('total_ead'))}",
                     f"Observed default rate (historical): {_pct(kpis.get('observed_default_rate'))}",
                     f"Average predicted PD: {_pct(kpis.get('avg_pd'))}",
                     f"Expected credit loss: {_fmt(kpis.get('total_el'))} ({_pct(kpis.get('el_rate'))} of EAD)",
                     f"High-risk share (count): {_pct(kpis.get('high_risk_share_count'))}"],
            tables=[("By risk band", ctx["band_table"])] if ctx.get("band_table") is not None else []))
    if ctx.get("contribution"):
        c = ctx["contribution"]
        s.append(ReportSection("Simplified lending economics", "SIMULATED", bullets=[
            f"{k.replace('_', ' ')}: {_pct(v) if 'rate' in k or 'margin' in k or k.endswith('_pd') else _fmt(v)}"
            for k, v in c.items()]))
    if ctx.get("scenarios") is not None:
        s.append(ReportSection("Stress test (illustrative scenarios)", "SIMULATED",
                               paragraphs=["Scenario multipliers are user-defined and illustrative unless supported by external evidence."],
                               tables=[("Scenario results", ctx["scenarios"])]))
    if ctx.get("policies") is not None:
        s.append(ReportSection("Lending policy comparison (test set)", "SIMULATED",
                               paragraphs=["Outcome counts are measured on held-out data; costs and contributions apply the "
                                           "user's cost and rate assumptions. No threshold is universally optimal."],
                               tables=[("Policies", ctx["policies"])]))
    if ctx.get("fairness") is not None:
        s.append(ReportSection(f"Group-level validation by {ctx.get('fairness_attribute')}", "MEASURED",
                               paragraphs=["Audit only — the attribute is not a model input. One metric cannot establish fairness."],
                               tables=[("Group metrics", ctx["fairness"])]))
    s.append(ReportSection("Limitations", "CONTEXT", bullets=[
        "PD refers to the dataset's own target event and observation window; treating it as a 12-month PD is an assumption.",
        "PD × LGD × EAD is a simplified expected-loss estimate, not an IFRS 9 ECL or regulatory capital model.",
        "LGD and EAD are assumptions unless supplied from recovery data.",
        "Random splits estimate performance on similar applicants, not on future cohorts.",
        "Explanations show predictive association, not causation, and can be unstable with correlated features.",
        "Fairness checks are limited to the attributes available and do not establish legal compliance.",
        "Any real use would require independent model validation, governance approval and human review."]))
    return s


def _table_md(df: pd.DataFrame, max_rows: int = 40) -> str:
    df = df.head(max_rows)
    cols = [str(c) for c in df.columns]
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for row in df.itertuples(index=False):
        out.append("| " + " | ".join(_fmt(v).replace("|", "/") for v in row) + " |")
    return "\n".join(out)


def to_markdown(sections: list[ReportSection], title: str = "CreditIQ — Project Report") -> str:
    lines = [f"# {title}", "", f"_Generated {datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC} by CreditIQ._", ""]
    lines.append("> Educational decision-support prototype — not for real credit decisions.")
    lines.append("")
    for sec in sections:
        lines += [f"## {sec.title}", "", f"**Evidence type:** {KINDS.get(sec.kind, sec.kind)}", ""]
        lines += [p + "\n" for p in sec.paragraphs]
        lines += [f"- {b}" for b in sec.bullets]
        if sec.bullets:
            lines.append("")
        for caption, df in sec.tables:
            lines += [f"**{caption}**", "", _table_md(df), ""]
    return "\n".join(lines)


_CSS = """
body{font-family:-apple-system,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;color:#0f1b2d;max-width:1040px;margin:32px auto;padding:0 20px;line-height:1.5}
h1{color:#0b2545;border-bottom:3px solid #1f4e99;padding-bottom:8px}h2{color:#0b2545;margin-top:32px}
.kind{display:inline-block;font-size:12px;font-weight:600;letter-spacing:.04em;padding:2px 8px;border-radius:10px;margin-bottom:6px}
.MEASURED{background:#e3edfb;color:#1f4e99}.OBSERVED{background:#e6f4ef;color:#1d6b4f}.ASSUMPTION{background:#fdf1df;color:#8a5a00}
.SIMULATED{background:#f1e8fb;color:#5b3a8c}.CONTEXT{background:#eef1f5;color:#4a5568}
table{border-collapse:collapse;width:100%;font-size:13px;margin:8px 0 18px}th{background:#0b2545;color:#fff;text-align:left;padding:6px}
td{border-bottom:1px solid #dde3ec;padding:5px 6px}tr:nth-child(even) td{background:#f7f9fc}.note{color:#5a6b85;font-size:13px}
"""


def to_html(sections: list[ReportSection], title: str = "CreditIQ — Project Report") -> str:
    e = html.escape
    parts = [f"<!doctype html><html><head><meta charset='utf-8'><title>{e(title)}</title><style>{_CSS}</style></head><body>",
             f"<h1>{e(title)}</h1>",
             f"<p class='note'>Generated {datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC}. Educational decision-support prototype — not for real credit decisions.</p>"]
    for sec in sections:
        parts.append(f"<h2>{e(sec.title)}</h2><span class='kind {e(sec.kind)}'>{e(KINDS.get(sec.kind, sec.kind))}</span>")
        parts += [f"<p>{e(p)}</p>" for p in sec.paragraphs]
        if sec.bullets:
            parts.append("<ul>" + "".join(f"<li>{e(b)}</li>" for b in sec.bullets) + "</ul>")
        for caption, df in sec.tables:
            head = "".join(f"<th>{e(str(c))}</th>" for c in df.columns)
            body = "".join("<tr>" + "".join(f"<td>{e(_fmt(v))}</td>" for v in row) + "</tr>" for row in df.head(40).itertuples(index=False))
            parts.append(f"<p><strong>{e(caption)}</strong></p><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>")
    parts.append("</body></html>")
    return "".join(parts)


def model_card_markdown(meta: dict, comparison: pd.DataFrame | None = None) -> str:
    """Model card generated from the actual experiment metadata (no placeholder metrics)."""
    key = meta.get("bundle_model") or meta.get("selected_model")
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
             f"{_fmt(ds.get('n_rows_loaded'))} rows loaded.",
             f"- Target: `{tgt.get('column')}` with default label(s) {tgt.get('positive_values (default)')}; "
             f"prevalence {_pct(tgt.get('prevalence'))}.",
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
             f"- Decision threshold: {_pct(model.get('threshold'))} ({cfg.get('threshold_rule')} on validation).", "",
             "## Validation (held-out test split)",
             f"- Split: {split.get('strategy')}; sizes {split.get('sizes')}.",
             f"- ROC-AUC {_fmt(test.get('roc_auc'))}, Gini {_fmt(test.get('gini'))}, PR-AUC {_fmt(test.get('pr_auc'))} "
             f"(prevalence {_pct(test.get('prevalence'))}), Brier {_fmt(test.get('brier'))}, KS {_fmt(test.get('ks'))}.",
             f"- At the threshold: recall {_pct(test.get('recall'))}, precision {_pct(test.get('precision'))}, "
             f"specificity {_pct(test.get('specificity'))}, flagged {_pct(test.get('flag_rate'))} of applicants.",
             ]
    if ds.get("is_synthetic"):
        lines.append("- These figures come from synthetic data and are not evidence of real-world performance.")
    if comparison is not None and len(comparison):
        lines += ["", "### Comparison with other trained models", "",
                  _table_md(comparison[["model", "val_roc_auc", "test_roc_auc", "test_pr_auc", "test_brier"]])]
    lines += ["", "## Limitations",
              "- PD refers to the dataset's own target event and window; mapping it to a 12-month PD is an assumption.",
              "- Explanations are associational, not causal; attributions are unstable for correlated inputs.",
              "- Random splits do not test performance on future cohorts.", "",
              "## Fairness considerations",
              "Sensitive attributes are excluded from inputs by default but remaining inputs can act as proxies. Group-level "
              "error rates and calibration should be reviewed on the Explainability & Responsible AI page; no single metric "
              "establishes fairness, and legal requirements vary by jurisdiction.", "",
              "## Governance requirements before any real use",
              "- Independent validation including out-of-time testing and benchmarking.",
              "- Legal review of inputs and adverse-action reasoning.",
              "- Human review of flagged and declined applications; documented override process.",
              "- Ongoing monitoring of drift, calibration and group outcomes with defined escalation thresholds."]
    return "\n".join(lines) + "\n"
