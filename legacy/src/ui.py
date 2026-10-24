"""Shared UI kit: styling, provenance-tagged metric tiles, formatting, chart and table helpers.

Design tokens
-------------
ink #12233D (text) · ledger blue #1E4C8A (primary) · steel #5A6B82 (secondary
text) · paper #F5F7FA (background) · rule #D9E1EB (borders).

Every headline number carries a provenance tag so the reader always knows
whether it is an observed fact, a model output, a user assumption or a
simulation — the core honesty requirement of the project.
"""
from __future__ import annotations

import html
import math
from typing import Iterable

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from . import config, state

INK, LEDGER, STEEL, PAPER, RULE = "#12233D", "#1E4C8A", "#5A6B82", "#F5F7FA", "#D9E1EB"
DEFAULT_COLOR, REPAID_COLOR = "#A23B3B", "#7C93B3"
MODEL_COLORS = {"logreg": "#12233D", "random_forest": "#4E7FA8", "gbm": "#C2873A"}
BAND_COLORS = {"Low": "#2F6B5E", "Moderate": "#B9A04E", "Elevated": "#C9783A", "High": "#A23B3B"}

PROVENANCE = {
    "observed": ("Observed", "#2F6B5E", "Historical fact from the loaded data"),
    "measured": ("Measured on held-out data", "#1E4C8A", "Computed by evaluating the model on data it did not train on"),
    "model": ("Model output", "#1E4C8A", "Produced by the trained model"),
    "assumption": ("Assumption", "#9A6B1F", "Value entered by the user"),
    "simulated": ("Simulated", "#5D4A8A", "Projection combining model outputs with assumptions"),
    "synthetic": ("Synthetic data", "#A23B3B", "Computed on fabricated demonstration data"),
}

_CSS = f"""
<style>
:root {{ --ink:{INK}; --ledger:{LEDGER}; --steel:{STEEL}; --rule:{RULE}; }}
.block-container {{ padding-top: 2.2rem; max-width: 1320px; }}
[data-testid="stMetricValue"], .ciq-tile-value, td, .ciq-num {{ font-variant-numeric: tabular-nums; }}
.ciq-header {{ margin: 0 0 1.1rem 0; padding-bottom: .9rem; border-bottom: 1px solid var(--rule); }}
.ciq-header h1 {{ font-family: 'Source Serif 4', Georgia, serif; font-weight: 600; font-size: 2.05rem;
  line-height: 1.15; color: var(--ink); margin: 0 0 .35rem 0; padding: 0; letter-spacing: -.01em; }}
.ciq-header p {{ color: var(--steel); font-size: 1rem; margin: 0; max-width: 78ch; }}
.ciq-tiles {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 10px; margin: .2rem 0 1rem 0; }}
.ciq-tile {{ background: #fff; border: 1px solid var(--rule); border-left: 3px solid var(--accent, var(--ledger));
  border-radius: 4px; padding: 11px 14px 10px 14px; min-height: 96px; }}
.ciq-tile-label {{ color: var(--steel); font-size: .84rem; font-weight: 500; line-height: 1.25; }}
.ciq-tile-label[title] {{ text-decoration: underline dotted #9AA8BC; text-underline-offset: 3px; cursor: help; }}
.ciq-tile-value {{ color: var(--ink); font-size: 1.62rem; font-weight: 600; line-height: 1.25; margin: 4px 0 3px 0; }}
.ciq-tile-foot {{ font-size: .76rem; color: var(--steel); line-height: 1.3; }}
.ciq-prov {{ color: var(--accent, var(--ledger)); font-weight: 600; }}
.ciq-banner {{ border: 1px solid #E7C9C9; background: #FBF3F3; color: #7A2E2E; border-radius: 4px;
  padding: 7px 12px; font-size: .86rem; margin-bottom: .9rem; }}
.ciq-banner b {{ color: #A23B3B; }}
.ciq-empty {{ border: 1px dashed #B8C4D4; border-radius: 6px; padding: 22px 24px; background: #fff; max-width: 760px; }}
.ciq-empty h3 {{ font-family: 'Source Serif 4', Georgia, serif; margin: 0 0 .3rem 0; color: var(--ink); font-size: 1.3rem; }}
.ciq-empty p {{ color: var(--steel); margin: 0; }}
.ciq-note {{ color: var(--steel); font-size: .86rem; line-height: 1.45; }}
.ciq-code {{ font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: .82em; background: #E9EEF4;
  color: var(--ink); padding: 1px 5px; border-radius: 3px; white-space: nowrap; }}
.ciq-steps {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 0; margin: .4rem 0 1.2rem 0;
  border: 1px solid var(--rule); border-radius: 6px; background: #fff; overflow: hidden; }}
.ciq-step {{ padding: 12px 14px; border-right: 1px solid var(--rule); position: relative; }}
.ciq-step:last-child {{ border-right: none; }}
.ciq-step .n {{ font-family: 'Source Serif 4', Georgia, serif; font-size: 1.35rem; color: #9AA8BC; line-height: 1; }}
.ciq-step .t {{ font-weight: 600; color: var(--ink); margin-top: 6px; font-size: .95rem; }}
.ciq-step .s {{ color: var(--steel); font-size: .8rem; margin-top: 2px; }}
.ciq-step.done {{ box-shadow: inset 0 -3px 0 #2F6B5E; }}
.ciq-step.done .n {{ color: #2F6B5E; }}
.ciq-step.next {{ box-shadow: inset 0 -3px 0 var(--ledger); background: #F3F7FC; }}
.ciq-step.next .n {{ color: var(--ledger); }}
.ciq-side {{ font-size: .82rem; line-height: 1.45; margin-top: .4rem; }}
.ciq-side .k {{ color: #8FA4C2; font-size: .74rem; margin-top: 8px; }}
.ciq-side .v {{ color: #E8EEF6; word-break: break-word; }}
.ciq-side .tag {{ display: inline-block; margin-top: 3px; padding: 0 6px; border-radius: 3px; font-size: .72rem;
  background: #5B2A33; color: #F6DADA; }}
.ciq-footer {{ margin-top: 2.5rem; padding-top: .8rem; border-top: 1px solid var(--rule); color: var(--steel); font-size: .78rem; }}
.ciq-legend {{ font-size: .8rem; color: var(--steel); margin: -.4rem 0 .8rem 0; }}
.ciq-legend span {{ margin-right: 14px; white-space: nowrap; }}
.ciq-legend i {{ display: inline-block; width: 9px; height: 9px; border-radius: 2px; margin-right: 5px; vertical-align: 0; }}
.ciq-callout {{ border-left: 3px solid var(--ledger); background: #fff; padding: 10px 14px; border-radius: 0 4px 4px 0;
  color: var(--ink); font-size: .92rem; margin: .3rem 0 1rem 0; }}
.ciq-callout.warn {{ border-left-color: #B7832F; }}
.ciq-callout.risk {{ border-left-color: #A23B3B; }}
.ciq-callout.ok {{ border-left-color: #2F6B5E; }}
@media (max-width: 640px) {{ .ciq-header h1 {{ font-size: 1.6rem; }} .ciq-tile-value {{ font-size: 1.35rem; }} }}
</style>
"""


def inject_css() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def _bad(v) -> bool:
    return v is None or (isinstance(v, (float, np.floating)) and not math.isfinite(v))


def pct(v, digits: int = 1) -> str:
    return "n/a" if _bad(v) else f"{v * 100:.{digits}f}%"


def num(v, digits: int = 0) -> str:
    return "n/a" if _bad(v) else f"{v:,.{digits}f}"


def dec(v, digits: int = 3) -> str:
    return "n/a" if _bad(v) else f"{v:.{digits}f}"


def val(v) -> str:
    """Readable value for an input shown next to an explanation."""
    if isinstance(v, str):
        return v
    if _bad(v) or (isinstance(v, float) and np.isnan(v)):
        return "missing"
    v = float(v)
    if abs(v) >= 1000 or v == int(v):
        return f"{v:,.0f}"
    return f"{v:.4g}"


def money(v, compact: bool = True) -> str:
    if _bad(v):
        return "n/a"
    cur = currency_prefix()
    sign = "−" if v < 0 else ""
    a = abs(float(v))
    if compact:
        for div, suf in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")):
            if a >= div:
                return f"{sign}{cur}{a / div:,.2f}{suf}"
    return f"{sign}{cur}{a:,.0f}"


def currency_prefix() -> str:
    """Symbol shown before amounts: a one-character symbol is attached ('₹' → '₹1.2M'); codes and abbreviations get a
    space ('INR' → 'INR 1.2M', 'Rs.' → 'Rs. 1.2M')."""
    cur = (state.currency() if "fin" in st.session_state else "").strip()
    return f"{cur} " if cur and (len(cur) > 1 or cur.isalpha()) else cur


def amount_label(what: str = "Amount") -> str:
    """Axis title for money values, using the chosen currency (or 'currency units' when none is set)."""
    cur = (state.currency() if "fin" in st.session_state else "").strip()
    return f"{what} ({cur})" if cur else f"{what} (currency units)"


# ---------------------------------------------------------------------------
# Layout components
# ---------------------------------------------------------------------------

def page_header(title: str, subtitle: str = "") -> None:
    sub = f"<p>{html.escape(subtitle)}</p>" if subtitle else ""
    st.markdown(f"<div class='ciq-header'><h1>{html.escape(title)}</h1>{sub}</div>", unsafe_allow_html=True)


def tile(label: str, value: str, prov: str | None = None, help: str | None = None, sub: str | None = None) -> str:
    name, color, prov_help = PROVENANCE.get(prov or "", (None, LEDGER, None))
    title_attr = f" title='{html.escape(help, quote=True)}'" if help else ""
    foot = []
    if name:
        foot.append(f"<span class='ciq-prov' title='{html.escape(prov_help, quote=True)}'>{html.escape(name)}</span>")
    if sub:
        foot.append(html.escape(sub))
    return (f"<div class='ciq-tile' style='--accent:{color}'><div class='ciq-tile-label'{title_attr}>{html.escape(label)}</div>"
            f"<div class='ciq-tile-value'>{html.escape(value)}</div><div class='ciq-tile-foot'>{'<br>'.join(foot)}</div></div>")


def tiles(items: Iterable[dict]) -> None:
    """Render a responsive row of metric tiles. Each item: label, value, prov, help, sub."""
    body = "".join(tile(**it) for it in items)
    st.markdown(f"<div class='ciq-tiles'>{body}</div>", unsafe_allow_html=True)


def provenance_legend(kinds: Iterable[str]) -> None:
    spans = "".join(f"<span><i style='background:{PROVENANCE[k][1]}'></i>{PROVENANCE[k][0]}: {PROVENANCE[k][2].lower()}</span>" for k in kinds)
    st.markdown(f"<div class='ciq-legend'>{spans}</div>", unsafe_allow_html=True)


def callout(text: str, kind: str = "info") -> None:
    cls = {"info": "", "warn": "warn", "risk": "risk", "ok": "ok"}.get(kind, "")
    st.markdown(f"<div class='ciq-callout {cls}'>{text}</div>", unsafe_allow_html=True)


def note(text: str) -> None:
    st.markdown(f"<div class='ciq-note'>{text}</div>", unsafe_allow_html=True)


def empty_state(title: str, body: str, page: str | None = None, link_label: str | None = None) -> None:
    st.markdown(f"<div class='ciq-empty'><h3>{html.escape(title)}</h3><p>{body}</p></div>", unsafe_allow_html=True)
    if page:
        st.write("")
        st.page_link(page, label=link_label or "Open", icon=":material/arrow_forward:")


def calibration_warning(bundle) -> None:
    """Shown wherever PDs feed money figures, if the active model's outputs are not calibrated."""
    model = getattr(bundle, "model", bundle)  # accepts a ModelBundle or a CreditModel
    if model is not None and not model.is_calibrated:
        callout("<b>The active model is not calibrated.</b> Its outputs are ranking scores, not probabilities of default, so "
                "expected-loss and contribution figures built on them are distorted. Retrain with Platt or isotonic calibration.",
                "risk")


def synthetic_banner() -> None:
    info = st.session_state.get("ds_info")
    if info is not None and info.is_synthetic:
        st.markdown("<div class='ciq-banner'><b>Synthetic demonstration data.</b> Every number on this page is computed "
                    "from fabricated records and says nothing about real borrowers or real lending performance.</div>",
                    unsafe_allow_html=True)


def flash(message: str, kind: str = "success") -> None:
    """Queue a message to show after the next rerun (so the sidebar refreshes too)."""
    st.session_state["_flash"] = (kind, message)


def show_flash() -> None:
    """Show a queued message as a toast (does not shift the page layout, so open tabs stay open)."""
    item = st.session_state.pop("_flash", None)
    if item:
        kind, message = item
        st.toast(message, icon={"success": ":material/check_circle:", "warning": ":material/warning:"}.get(kind, ":material/info:"))


def tabs(labels: list[str], key: str):
    """Tabs that remember the selected tab across reruns where Streamlit supports it."""
    try:
        return st.tabs(labels, key=key)
    except TypeError:  # older Streamlit without stateful tabs
        return st.tabs(labels)


def footer() -> None:
    st.markdown(f"<div class='ciq-footer'>{html.escape(config.APP_DISCLAIMER)} All data and models stay on this computer.</div>",
                unsafe_allow_html=True)


def steps(items: list[tuple[str, str, str]]) -> None:
    """Workflow strip. items: (title, subtitle, status in {'done','next','todo'})."""
    cells = "".join(f"<div class='ciq-step {s}'><div class='n'>{i}</div><div class='t'>{html.escape(t)}</div>"
                    f"<div class='s'>{html.escape(sub)}</div></div>" for i, (t, sub, s) in enumerate(items, start=1))
    st.markdown(f"<div class='ciq-steps'>{cells}</div>", unsafe_allow_html=True)


def pct_input(container, label: str, value: float, max_pct: float = 100.0, step: float = 0.5, digits: int = 2,
              help: str | None = None, key: str | None = None) -> float:
    """Number input shown in percent; returns a fraction (e.g. 45 → 0.45)."""
    shown = container.number_input(f"{label} (%)", min_value=0.0, max_value=float(max_pct), value=round(float(value) * 100, digits),
                                   step=float(step), format=f"%.{digits}f", help=help, key=key)
    return float(shown) / 100.0


def pick(widget, label: str, options: dict, default=None, **kwargs):
    """Selectbox/radio over a {value: display label} mapping; returns the value.

    Passing display labels as the options (instead of ``format_func``) keeps
    widget state stable and testable.
    """
    keys = list(options)
    labels = [str(options[k]) for k in keys]
    index = keys.index(default) if default in keys else 0
    chosen = widget(label, labels, index=index, **kwargs)
    return keys[labels.index(chosen)] if chosen in labels else keys[index]


def pick_many(widget, label: str, options: dict, default=None, **kwargs) -> list:
    keys = list(options)
    labels = [str(options[k]) for k in keys]
    chosen = widget(label, labels, default=[options[k] for k in (default or []) if k in options], **kwargs)
    return [keys[labels.index(c)] for c in chosen]


# ---------------------------------------------------------------------------
# Guards
# ---------------------------------------------------------------------------

def require_data() -> None:
    if not state.has_data():
        empty_state("No dataset loaded yet", "Load the bundled synthetic sample, upload a CSV or open a local file to begin.",
                    "app_pages/2_Data_Center.py", "Open the Data Center")
        st.stop()


def require_target() -> None:
    require_data()
    if not state.has_target():
        empty_state("Confirm the target first", "Choose the outcome column and confirm which label means default in the Data Center.",
                    "app_pages/2_Data_Center.py", "Map the target")
        st.stop()


def require_model() -> None:
    if not state.has_model():
        empty_state("No trained model", "Train models in the AI Model Lab, or load a saved model bundle there.",
                    "app_pages/4_Model_Lab.py", "Open the AI Model Lab")
        st.stop()


def require_experiment(what: str = "this analysis") -> None:
    if not state.has_experiment():
        if state.has_model():
            empty_state("Needs the training session", f"{what.capitalize()} uses the held-out split from a training run in this session. "
                        "A model loaded from disk does not carry borrower rows. Train a model in the AI Model Lab to use it.",
                        "app_pages/4_Model_Lab.py", "Open the AI Model Lab")
        else:
            require_model()
        st.stop()


# ---------------------------------------------------------------------------
# Charts and tables
# ---------------------------------------------------------------------------

SANS = "'IBM Plex Sans', 'Segoe UI', Helvetica, Arial, sans-serif"
SERIF = "'Source Serif 4', Georgia, 'Times New Roman', serif"
COLORWAY = ["#1E4C8A", "#C2873A", "#4E7FA8", "#2F6B5E", "#A23B3B", "#5D4A8A", "#8A9BB3"]


def _has_legend(fig: go.Figure) -> bool:
    if fig.layout.showlegend is False:
        return False
    shown = [t for t in fig.data if t.showlegend is not False and getattr(t, "type", "") not in ("heatmap", "waterfall")]
    return len(shown) > 1


def style_figure(fig: go.Figure, height: int = 360) -> go.Figure:
    """Apply the CreditIQ chart style as explicit layout values (titles, axis formats set earlier are kept)."""
    axis = dict(gridcolor="#E8EDF3", linecolor="#C9D3E0", zeroline=False, tickcolor="#C9D3E0",
                title_font=dict(size=12, color=STEEL), tickfont=dict(size=11, color=STEEL))
    fig.update_layout(
        template="plotly_white", height=height, paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF",
        font=dict(family=SANS, size=12, color=INK), colorway=COLORWAY,
        margin=dict(l=14, r=14, t=86 if _has_legend(fig) else 58, b=14),
        title=dict(font=dict(family=SERIF, size=16, color=INK), x=0.012, xanchor="left", y=0.975, yanchor="top"),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0, font=dict(size=11, color=STEEL),
                    bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(bgcolor="#FFFFFF", bordercolor=RULE, font=dict(size=12, color=INK)),
        bargap=0.25,
    )
    fig.update_xaxes(**axis)
    fig.update_yaxes(**axis)
    return fig


def plot(fig: go.Figure, height: int = 360, key: str | None = None) -> None:
    style_figure(fig, height)
    try:
        st.plotly_chart(fig, theme=None, width="stretch", key=key, config={"displaylogo": False})
    except TypeError:  # older Streamlit
        st.plotly_chart(fig, theme=None, use_container_width=True, key=key, config={"displaylogo": False})


_TOKENS = {"pd": "PD", "ead": "EAD", "lgd": "LGD", "el": "EL", "roc": "ROC", "auc": "AUC", "pr": "PR", "ks": "KS",
           "tpr": "TPR", "fpr": "FPR", "fnr": "FNR", "hhi": "HHI", "psi": "PSI", "ece": "ECE", "f1": "F1", "id": "ID",
           "ci": "CI", "brier": "Brier", "gini": "Gini", "tp": "TP", "fp": "FP", "tn": "TN", "fn": "FN", "val": "Validation", "dr": "DR", "p1": "P1", "p99": "P99"}


def pretty(name) -> str:
    """'test_roc_auc' → 'Test ROC AUC'. Column names that are not snake_case are left as they are."""
    text = str(name)
    if text.upper() == text or " " in text or "_" not in text and text[:1].isupper():
        return text
    words = [_TOKENS.get(w.lower(), w) for w in text.split("_")]
    out = " ".join(words)
    return out[:1].upper() + out[1:]


def show_table(df: pd.DataFrame, pct_cols: Iterable[str] = (), money_cols: Iterable[str] = (),
               dec_cols: Iterable[str] = (), int_cols: Iterable[str] = (), height: int | None = None,
               hide_index: bool = True, pretty_headers: bool = True) -> None:
    fmt = {}
    for c in pct_cols:
        if c in df.columns:
            fmt[c] = lambda v: pct(v, 2)
    for c in money_cols:
        if c in df.columns:
            fmt[c] = lambda v: money(v, compact=False)
    for c in dec_cols:
        if c in df.columns:
            fmt[c] = lambda v: dec(v, 4)
    for c in int_cols:
        if c in df.columns:
            fmt[c] = lambda v: num(v, 0)
    data = df
    if pretty_headers:
        names = {c: pretty(c) for c in df.columns}
        if len(set(names.values())) == len(names):
            data = df.rename(columns=names)
            fmt = {names[c]: f for c, f in fmt.items()}
    data = data.style.format(fmt, na_rep="n/a") if fmt else data
    kwargs = {"hide_index": hide_index}
    natural = 35 * len(df) + 40
    if height or len(df) > 12:
        kwargs["height"] = int(min(height or 420, natural))
    try:
        st.dataframe(data, width="stretch", **kwargs)
    except TypeError:
        st.dataframe(data, use_container_width=True, **kwargs)


def download(label: str, data, file_name: str, mime: str, key: str | None = None, help: str | None = None) -> None:
    from .data_loader import sanitize_filename

    kwargs = dict(data=data, file_name=sanitize_filename(file_name), mime=mime, key=key, help=help, icon=":material/download:")
    try:
        # "ignore": downloading must not rerun the page (a rerun recomputes heavy pages and can race the download)
        st.download_button(label, on_click="ignore", **kwargs)
    except TypeError:  # Streamlit without on_click="ignore"
        st.download_button(label, **kwargs)


def sidebar_status() -> None:
    ss = st.session_state
    parts = ["<div class='ciq-side'>"]
    info = ss.get("ds_info")
    if info is None:
        parts.append("<div class='k'>Dataset</div><div class='v'>None loaded</div>")
    else:
        parts.append(f"<div class='k'>Dataset</div><div class='v'>{html.escape(info.name)}<br>{info.n_rows_loaded:,} rows</div>")
        if info.is_synthetic:
            parts.append("<span class='tag'>Synthetic</span>")
    tgt = ss.get("target")
    if tgt is not None:
        labels = ", ".join(map(str, tgt.positive_values))
        parts.append(f"<div class='k'>Target</div><div class='v'>{html.escape(tgt.column)} = {html.escape(labels)}<br>"
                     f"{pct(tgt.prevalence)} default rate</div>")
    bundle = state.active_bundle()
    if bundle is not None:
        src = "loaded from disk" if ss.get("use_loaded_bundle") or ss.get("experiment") is None else "this session"
        auc = bundle.metadata.get("metrics", {}).get(bundle.model.key, {}).get("test", {}).get("roc_auc")
        parts.append(f"<div class='k'>Active model ({src})</div><div class='v'>{html.escape(bundle.model.label)}"
                     f"<br>Test ROC-AUC {dec(auc, 3)}<br>Threshold {pct(state.active_threshold())}</div>")
    else:
        parts.append("<div class='k'>Model</div><div class='v'>Not trained</div>")
    parts.append("</div>")
    st.sidebar.markdown("".join(parts), unsafe_allow_html=True)
