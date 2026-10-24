"""Data profiling, quality checks, target mapping and leakage screening.

Nothing here silently changes the data. Functions return findings and
*suggested* roles; the user confirms them in the Data Center.
"""
from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from creditiq_core._frames import iter_rows

# ---------------------------------------------------------------------------
# Column typing
# ---------------------------------------------------------------------------

KIND_NUMERIC = "numeric"
KIND_CATEGORICAL = "categorical"
KIND_DATETIME = "datetime"
KIND_EMPTY = "empty"


def infer_kind(series: pd.Series) -> str:
    """Classify a column as numeric, categorical, datetime or empty (all missing)."""
    if series.isna().all():
        return KIND_EMPTY
    if pd.api.types.is_bool_dtype(series):
        return KIND_CATEGORICAL
    if pd.api.types.is_numeric_dtype(series):
        return KIND_NUMERIC
    if pd.api.types.is_datetime64_any_dtype(series):
        return KIND_DATETIME
    return KIND_CATEGORICAL


def _looks_like_date(series: pd.Series, name: str) -> bool:
    if pd.api.types.is_datetime64_any_dtype(series):
        return True
    if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
        return False
    hint = re.search(r"(date|_dt$|_d$|time|month|period|issue)", name, flags=re.I)
    if not hint:
        return False
    sample = series.dropna().astype(str).head(500)
    if sample.empty:
        return False
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
        except (TypeError, ValueError):
            parsed = pd.to_datetime(sample, errors="coerce")
    return parsed.notna().mean() >= 0.9


def detect_date_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if _looks_like_date(df[c], str(c))]


def parse_dates(series: pd.Series) -> pd.Series:
    """Parse a date column robustly (NaT on failure)."""
    if pd.api.types.is_datetime64_any_dtype(series):
        return series
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            return pd.to_datetime(series, errors="coerce", format="mixed")
        except (TypeError, ValueError):
            return pd.to_datetime(series, errors="coerce")


# ---------------------------------------------------------------------------
# Profiling
# ---------------------------------------------------------------------------

def profile_dataframe(df: pd.DataFrame) -> tuple[dict[str, Any], pd.DataFrame]:
    """Return (summary dict, per-column profile table)."""
    if df is None or df.shape[1] == 0:
        raise ValueError("Dataset is empty (no columns).")
    n = len(df)
    rows = []
    for col in df.columns:
        s = df[col]
        kind = infer_kind(s)
        n_missing = int(s.isna().sum())
        try:
            n_unique = int(s.nunique(dropna=True))
        except TypeError:
            n_unique = int(s.astype(str).nunique(dropna=True))
        row: dict[str, Any] = {
            "column": col,
            "dtype": str(s.dtype),
            "kind": kind,
            "missing": n_missing,
            "missing_pct": n_missing / n if n else 0.0,
            "unique": n_unique,
            "unique_ratio": n_unique / max(n - n_missing, 1),
            "n_inf": 0,
            "min": np.nan,
            "median": np.nan,
            "max": np.nan,
            "example": "",
        }
        if kind == KIND_NUMERIC:
            vals = pd.to_numeric(s, errors="coerce")
            finite = vals.replace([np.inf, -np.inf], np.nan)
            row["n_inf"] = int(np.isinf(vals.to_numpy(dtype=float, na_value=np.nan)).sum())
            if finite.notna().any():
                row["min"] = float(finite.min())
                row["median"] = float(finite.median())
                row["max"] = float(finite.max())
        non_null = s.dropna()
        if len(non_null):
            row["example"] = ", ".join(map(str, non_null.astype(str).value_counts().head(3).index))
        rows.append(row)
    prof = pd.DataFrame(rows)

    try:
        dup_rows = int(df.duplicated().sum()) if n <= 2_000_000 else -1
    except TypeError:
        dup_rows = int(df.astype(str).duplicated().sum())
    summary = {
        "rows": n,
        "columns": df.shape[1],
        "numeric_columns": int((prof["kind"] == KIND_NUMERIC).sum()),
        "categorical_columns": int((prof["kind"] == KIND_CATEGORICAL).sum()),
        "missing_cells_pct": float(df.isna().to_numpy().mean()) if n else 0.0,
        "duplicate_rows": dup_rows,
        "memory_mb": float(df.memory_usage(deep=True).sum() / 1e6),
    }
    return summary, prof


SENTINEL_VALUES = (365243, 999999, 99999, -999, -9999, 9999999)


def outlier_review(df: pd.DataFrame, max_cols: int = 200) -> pd.DataFrame:
    """IQR-based outlier counts and known sentinel values per numeric column."""
    rows = []
    for col in df.columns[:max_cols * 3]:
        s = df[col]
        if infer_kind(s) != KIND_NUMERIC:
            continue
        v = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        if len(v) < 10:
            continue
        q1, q3 = v.quantile([0.25, 0.75])
        iqr = q3 - q1
        lo, hi = q1 - 3 * iqr, q3 + 3 * iqr
        n_out = int(((v < lo) | (v > hi)).sum()) if iqr > 0 else 0
        sentinels = {int(x): int((v == x).sum()) for x in SENTINEL_VALUES if (v == x).sum() > 0}
        if n_out or sentinels:
            rows.append({
                "column": col,
                "extreme_values_(3xIQR)": n_out,
                "extreme_pct": n_out / len(v),
                "sentinel_values": ", ".join(f"{k} (n={c:,})" for k, c in sentinels.items()) or "—",
                "p1": float(v.quantile(0.01)),
                "p99": float(v.quantile(0.99)),
            })
        if len(rows) >= max_cols:
            break
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Issues and suggested roles
# ---------------------------------------------------------------------------

@dataclass
class Issue:
    severity: str  # "error" | "warning" | "info"
    column: str
    category: str
    message: str

    def to_dict(self) -> dict[str, str]:
        return {"severity": self.severity, "column": self.column, "category": self.category, "message": self.message}


ID_NAME_PATTERN = re.compile(
    r"(^id$|_id$|^id_|sk_id|uuid|guid|^index$|^unnamed|member_id|loan_id|customer_id|"
    r"application_id|account_id|^url$|^desc$|emp_title|^name$|phone_number|email|zip_code|^zip$)",
    re.I,
)

SENSITIVE_PATTERN = re.compile(
    r"(gender|(^|_)sex($|_)|(^|_)race($|_)|ethnic|religio|nationality|national_origin|"
    r"citizenship|marital|family_status|personal_status|days_birth|(^|(?<!car)_)age($|_)|age_years|"
    r"date_of_birth|(^|_)dob($|_)|disab|pregnan|(^|_)caste($|_)|foreign_worker)",
    re.I,
)

# Columns documented as post-origination / post-outcome in common public credit datasets
# (e.g. LendingClub). These are excluded by default for application-time prediction.
KNOWN_POST_OUTCOME = {
    "total_pymnt", "total_pymnt_inv", "total_rec_prncp", "total_rec_int", "total_rec_late_fee",
    "recoveries", "collection_recovery_fee", "last_pymnt_d", "last_pymnt_amnt", "next_pymnt_d",
    "last_credit_pull_d", "out_prncp", "out_prncp_inv", "last_fico_range_high",
    "last_fico_range_low", "hardship_flag", "debt_settlement_flag", "settlement_status",
    "settlement_date", "settlement_amount", "settlement_percentage", "settlement_term",
    "pymnt_plan", "post_loan_collection_calls",
}

LEAKAGE_NAME_PATTERN = re.compile(
    r"(post_|_post|after|collection|recover|charge_?off|settle|write_?off|payoff|"
    r"last_pymnt|total_pymnt|total_rec|out_prncp|hardship|outcome|status$|closed|_paid|paid_)",
    re.I,
)


def find_issues(df: pd.DataFrame, profile: pd.DataFrame | None = None) -> list[Issue]:
    """Data-quality warnings that do not depend on the target."""
    issues: list[Issue] = []
    n = len(df)
    if n == 0:
        return [Issue("error", "—", "empty", "The dataset has no rows.")]
    if profile is None:
        _, profile = profile_dataframe(df)
    if n < 500:
        issues.append(Issue("warning", "—", "small_data", f"Only {n:,} rows. Model estimates will be unstable; treat results as illustrative."))
    for r in iter_rows(profile):
        if r.kind == KIND_EMPTY:
            issues.append(Issue("warning", r.column, "all_missing", "Column is entirely missing and will be excluded."))
            continue
        if r.unique <= 1:
            issues.append(Issue("info", r.column, "constant", "Column has a single value and carries no information."))
        if r.missing_pct > 0.6:
            issues.append(Issue("warning", r.column, "high_missing", f"{r.missing_pct:.0%} missing. Imputation and a missing-indicator will be used if included."))
        if r.n_inf:
            issues.append(Issue("warning", r.column, "infinite", f"{r.n_inf:,} infinite values; they are treated as missing during modelling."))
        if r.kind == KIND_CATEGORICAL and r.unique > 50:
            issues.append(Issue("info", r.column, "high_cardinality", f"{r.unique:,} distinct values; rare levels are grouped during encoding."))
        if r.kind == KIND_CATEGORICAL:
            sample = df[r.column].dropna().astype(str).head(1000)
            numeric_share = pd.to_numeric(sample, errors="coerce").notna().mean() if len(sample) else 0
            if numeric_share > 0.95 and r.unique > 20:
                issues.append(Issue("warning", r.column, "numeric_as_text", "Values look numeric but are stored as text; check the source formatting."))
    if (profile["column"] == "DAYS_EMPLOYED").any() and (pd.to_numeric(df["DAYS_EMPLOYED"], errors="coerce") == 365243).any():
        issues.append(Issue("info", "DAYS_EMPLOYED", "sentinel", "Value 365243 is a known Home Credit placeholder (mostly pensioners/unemployed). The feature-engineering step converts it to missing and adds a flag."))
    try:
        dups = int(df.duplicated().sum())
    except TypeError:
        dups = 0
    if dups:
        issues.append(Issue("warning", "—", "duplicates", f"{dups:,} exact duplicate rows. Duplicates can leak between train and test splits."))
    return issues


def suggest_target_column(df: pd.DataFrame) -> str | None:
    candidates = ["TARGET", "target", "default", "is_default", "Default", "loan_status", "bad",
                  "class", "Class", "label", "Risk", "credit_risk", "y"]
    for c in candidates:
        if c in df.columns:
            return c
    return None


def suggest_exposure_columns(df: pd.DataFrame) -> list[str]:
    pattern = re.compile(r"(amt_credit$|loan_amnt|funded_amnt$|credit_amount|loan_amount|exposure|principal|^amount$|credit_limit)", re.I)
    return [c for c in df.columns if pattern.search(str(c)) and infer_kind(df[c]) == KIND_NUMERIC]


def suggest_feature_roles(
    df: pd.DataFrame,
    target_col: str | None,
    date_col: str | None = None,
    profile: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Suggest which columns to use as predictors and why others are excluded."""
    if profile is None:
        _, profile = profile_dataframe(df)
    n = len(df)
    rows = []
    for r in iter_rows(profile):
        col = r.column
        lname = str(col).lower()
        role, reason, use = "predictor", "", True
        if col == target_col:
            role, reason, use = "target", "Selected target column", False
        elif col == date_col:
            role, reason, use = "date", "Used for chronological splitting and time trends, not as a predictor", False
        elif r.kind == KIND_EMPTY:
            role, reason, use = "empty", "All values missing", False
        elif r.unique <= 1:
            role, reason, use = "constant", "Single value", False
        elif lname in KNOWN_POST_OUTCOME:
            role, reason, use = "leakage", "Known post-origination/post-outcome field — unavailable at application time", False
        elif ID_NAME_PATTERN.search(lname) or (r.unique_ratio >= 0.98 and n >= 50 and (
                (r.kind == KIND_CATEGORICAL and r.unique > 200) or _looks_like_sequence(df[col]))):
            role, reason, use = "identifier", "Looks like a unique identifier or free text", False
        elif r.kind == KIND_CATEGORICAL and r.unique > 200 and r.unique_ratio > 0.3:
            role, reason, use = "free_text", "Very high-cardinality text; unlikely to generalise", False
        elif r.kind == KIND_DATETIME or _looks_like_date(df[col], str(col)):
            role, reason, use = "date", "Raw date — not used directly as a predictor", False
        elif SENSITIVE_PATTERN.search(lname):
            role, reason, use = "sensitive", "Potentially protected/sensitive attribute — excluded from the model by default; available for the fairness audit", False
        elif LEAKAGE_NAME_PATTERN.search(lname):
            role, reason, use = "review", "Name suggests possible post-outcome information — confirm it is known at application time", True
        rows.append({"column": col, "kind": r.kind, "use": use, "role": role, "reason": reason,
                     "missing_pct": r.missing_pct, "unique": r.unique})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Target mapping
# ---------------------------------------------------------------------------

class TargetMappingError(ValueError):
    """Raised when a target mapping cannot produce a usable binary label."""


@dataclass
class TargetMapping:
    column: str
    positive_values: list[Any]
    negative_values: list[Any]
    definition: str = ""
    n_positive: int = 0
    n_negative: int = 0
    n_unmapped: int = 0
    warnings: list[str] = field(default_factory=list)

    @property
    def prevalence(self) -> float:
        total = self.n_positive + self.n_negative
        return self.n_positive / total if total else float("nan")

    def to_dict(self) -> dict[str, Any]:
        return {
            "column": self.column,
            "positive_values (default)": [str(v) for v in self.positive_values],
            "negative_values (non-default)": [str(v) for v in self.negative_values],
            "definition": self.definition,
            "n_positive": self.n_positive,
            "n_negative": self.n_negative,
            "n_unmapped_dropped": self.n_unmapped,
            "prevalence": self.prevalence,
        }


def label_key(value: Any) -> str:
    """Canonical text form of a target label so that 1, 1.0, np.int64(1) and "1" all match."""
    if isinstance(value, (bool, np.bool_)):
        return str(bool(value))
    if isinstance(value, (int, float, np.integer, np.floating)) and np.isfinite(value) and float(value).is_integer():
        return str(int(value))
    if isinstance(value, str):
        text = value.strip()
        try:
            num = float(text)
            if np.isfinite(num) and num.is_integer() and text.replace(".", "", 1).lstrip("+-").isdigit():
                return str(int(num))
        except ValueError:
            pass
        return text
    return str(value)


def _as_key(values: Any) -> set[str]:
    return {label_key(v) for v in values}


def _looks_like_sequence(series: pd.Series) -> bool:
    """Integer column whose values are (nearly) unique AND packed into a tight range — typical of row IDs,
    not of amounts or day counts, which are spread over a wide range."""
    if not pd.api.types.is_integer_dtype(series) or pd.api.types.is_bool_dtype(series):
        return False
    v = series.dropna()
    if v.empty:
        return False
    span = float(v.max()) - float(v.min()) + 1
    return span <= 3 * v.nunique()


def apply_target_mapping(series: pd.Series, positive_values: Any, negative_values: Any) -> pd.Series:
    """Return a float Series: 1.0 = default, 0.0 = non-default, NaN = unmapped/missing."""
    pos, neg = _as_key(positive_values), _as_key(negative_values)
    keys = series.map(label_key)
    out = pd.Series(np.nan, index=series.index, dtype=float)
    out[keys.isin(pos)] = 1.0
    out[keys.isin(neg)] = 0.0
    out[series.isna()] = np.nan
    return out


def build_target_mapping(
    df: pd.DataFrame,
    column: str,
    positive_values: Any,
    negative_values: Any = None,
    definition: str = "",
    min_class_count: int = 5,
) -> TargetMapping:
    """Validate a user-chosen mapping. Raises TargetMappingError when unusable."""
    if df is None or len(df) == 0:
        raise TargetMappingError("The dataset is empty.")
    if column not in df.columns:
        raise TargetMappingError(f"Target column '{column}' is not in the dataset.")
    positive_values = list(positive_values or [])
    if not positive_values:
        raise TargetMappingError("Select at least one value that represents DEFAULT.")
    series = df[column]
    observed = _as_key(series.dropna().unique())
    unknown = _as_key(positive_values) - observed
    if unknown:
        raise TargetMappingError(f"Values not found in '{column}': {sorted(unknown)}")
    if negative_values is None:
        negative_values = [v for v in series.dropna().unique() if label_key(v) not in _as_key(positive_values)]
    negative_values = list(negative_values)
    overlap = _as_key(positive_values) & _as_key(negative_values)
    if overlap:
        raise TargetMappingError(f"Values cannot be both default and non-default: {sorted(overlap)}")
    if not negative_values:
        raise TargetMappingError("No values remain for the NON-DEFAULT class; the target would have one class.")

    y = apply_target_mapping(series, positive_values, negative_values)
    n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
    n_unmapped = int(y.isna().sum())
    if n_pos == 0 or n_neg == 0:
        raise TargetMappingError("After mapping, the target has only one class. Both default and non-default records are required.")
    mapping = TargetMapping(column, positive_values, negative_values, definition, n_pos, n_neg, n_unmapped)
    if n_unmapped:
        mapping.warnings.append(f"{n_unmapped:,} rows have missing or unmapped target values and will be excluded from modelling.")
    if min(n_pos, n_neg) < min_class_count:
        mapping.warnings.append(f"The minority class has only {min(n_pos, n_neg)} records — too few for reliable modelling.")
    if pd.api.types.is_numeric_dtype(series) and observed == {"1", "2"}:
        mapping.warnings.append("Labels are 1/2. In UCI Statlog German Credit, 1 = good and 2 = bad — check your source's documentation.")
    prevalence = n_pos / (n_pos + n_neg)
    if prevalence > 0.5:
        mapping.warnings.append(f"{prevalence:.0%} of records are mapped to DEFAULT. Check that the default label was not inverted.")
    return mapping


def suggest_positive_values(series: pd.Series) -> tuple[list[Any], str]:
    """Heuristic suggestion for the default label(s) plus an explanation. Always requires user confirmation."""
    values = list(series.dropna().unique())
    keys = _as_key(values)
    if keys <= {"0", "1", "0.0", "1.0"} and len(keys) == 2:
        one = [v for v in values if str(v) in {"1", "1.0"}]
        return one, "Binary 0/1 column. Many datasets use 1 = event, but this is NOT guaranteed — confirm with the data dictionary."
    if keys == {"1", "2"}:
        return [], "Labels 1/2 detected. In UCI Statlog German Credit, 2 = bad credit risk. Confirm with your source before mapping."
    bad_pat = re.compile(r"(charged off|default|^bad$|^yes$|^y$|^true$|written off)", re.I)
    suggested = [v for v in values if bad_pat.search(str(v)) and not re.search(r"fully paid|^good$", str(v), re.I)]
    if suggested:
        return suggested, "Suggested from label text. Outstanding loans (e.g. 'Current') have no final outcome and are usually excluded rather than mapped."
    return [], "No suggestion — select the label(s) that represent default for this dataset."


UNRESOLVED_PATTERN = re.compile(r"(current|grace|late \(|issued|in review|pending|ongoing)", re.I)


def suggest_negative_values(series: pd.Series, positive_values: Any) -> list[Any]:
    """All remaining labels except ones that look like unresolved loans (e.g. 'Current')."""
    pos = _as_key(positive_values)
    return [v for v in series.dropna().unique() if label_key(v) not in pos and not UNRESOLVED_PATTERN.search(str(v))]


# ---------------------------------------------------------------------------
# Statistical leakage screen
# ---------------------------------------------------------------------------

def single_feature_auc(x: pd.Series, y: pd.Series) -> float | None:
    """Direction-free AUC of one feature against the binary target (None if undefined)."""
    mask = y.notna()
    x, y = x[mask], y[mask].astype(int)
    if y.nunique() < 2 or len(y) < 20:
        return None
    if infer_kind(x) == KIND_NUMERIC:
        v = pd.to_numeric(x, errors="coerce").replace([np.inf, -np.inf], np.nan)
        if v.notna().sum() < 10:
            return None
        score = v.fillna(v.median())
    else:
        keys = x.astype(str).fillna("__missing__")
        if keys.nunique() > 200:
            return None
        # Two-fold target encoding to avoid in-sample optimism
        idx = np.arange(len(keys))
        fold = idx % 2
        score = pd.Series(np.nan, index=keys.index, dtype=float)
        for f in (0, 1):
            train = fold != f
            rates = y[train].groupby(keys[train]).mean()
            score[fold == f] = keys[fold == f].map(rates).astype(float).values
        score = score.fillna(y.mean())
    if score.nunique() < 2:
        return 0.5
    auc = roc_auc_score(y, score)
    return float(max(auc, 1 - auc))


def leakage_scan(
    df: pd.DataFrame,
    y: pd.Series,
    columns: list[str],
    sample_size: int = 20_000,
    flag_threshold: float = 0.90,
    seed: int = 42,
) -> pd.DataFrame:
    """Screen each candidate predictor for suspiciously strong single-feature separation."""
    data = df[columns].copy()
    data["__y__"] = y.values if isinstance(y, pd.Series) else y
    data = data[data["__y__"].notna()]
    if len(data) > sample_size:
        data = data.sample(sample_size, random_state=seed)
    rows = []
    for col in columns:
        auc = single_feature_auc(data[col], data["__y__"])
        if auc is None:
            flag = "not assessed"
        elif auc >= flag_threshold:
            flag = "SUSPICIOUS — possible leakage"
        elif auc >= 0.75:
            flag = "strong (plausible, but verify timing)"
        else:
            flag = "ok"
        rows.append({"column": col, "single_feature_auc": auc, "assessment": flag})
    out = pd.DataFrame(rows)
    return out.sort_values("single_feature_auc", ascending=False, na_position="last").reset_index(drop=True)


def quality_report(dataset_info: dict[str, Any], summary: dict[str, Any], profile: pd.DataFrame, issues: list[Issue],
                   target: dict[str, Any] | None, roles: pd.DataFrame | None) -> str:
    """Markdown data-quality report built only from computed results."""
    lines = ["# CreditIQ — Data Quality Report", ""]
    lines.append("> Section type: **Historical data observations** (computed from the loaded file).")
    lines.append("")
    lines.append("## Dataset")
    for k, v in dataset_info.items():
        lines.append(f"- **{k}**: {v}")
    if dataset_info.get("is_synthetic"):
        lines.append("- **NOTE**: This dataset is SYNTHETIC and not real borrower data.")
    lines.append("")
    lines.append("## Summary")
    for k, v in summary.items():
        lines.append(f"- **{k}**: {v:,.3f}" if isinstance(v, float) else f"- **{k}**: {v}")
    lines.append("")
    if target:
        lines.append("## Target mapping (user-confirmed)")
        for k, v in target.items():
            lines.append(f"- **{k}**: {v}")
        lines.append("")
    lines.append("## Issues")
    if issues:
        lines.append("| Severity | Column | Category | Message |")
        lines.append("|---|---|---|---|")
        for i in issues:
            lines.append(f"| {i.severity} | {i.column} | {i.category} | {i.message} |")
    else:
        lines.append("No issues detected by the automated checks.")
    lines.append("")
    if roles is not None and len(roles):
        excluded = roles[~roles["use"]]
        lines.append("## Excluded columns")
        if len(excluded):
            lines.append("| Column | Role | Reason |")
            lines.append("|---|---|---|")
            for r in iter_rows(excluded):
                lines.append(f"| {r.column} | {r.role} | {r.reason} |")
        else:
            lines.append("None.")
        lines.append("")
    lines.append("## Column profile")
    cols = ["column", "dtype", "kind", "missing_pct", "unique", "min", "median", "max"]
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("|" + "---|" * len(cols))
    for r in iter_rows(profile[cols], index=False):
        cells = []
        for c, v in zip(cols, r):
            if c == "missing_pct":
                cells.append(f"{v:.1%}")
            elif isinstance(v, float):
                cells.append("" if np.isnan(v) else f"{v:,.4g}")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"
