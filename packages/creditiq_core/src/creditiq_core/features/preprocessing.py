"""Train/validation/test splitting and the leakage-safe preprocessing pipeline.

All fitted preprocessing (imputation, scaling, encoding) sits inside a
scikit-learn ``Pipeline`` that is fitted on the TRAINING split only. Validation
and test rows are transformed with parameters learned from training rows.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from creditiq_core.data.validation import parse_dates


def select_numeric(X: pd.DataFrame) -> list[str]:
    """Column selector used by the ColumnTransformer (module-level so it pickles)."""
    return [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c]) and not pd.api.types.is_bool_dtype(X[c])]


def select_categorical(X: pd.DataFrame) -> list[str]:
    numeric = set(select_numeric(X))
    return [c for c in X.columns if c not in numeric]


def build_preprocessor(scale_numeric: bool = True, max_categories: int = 30, min_frequency: float = 0.002) -> ColumnTransformer:
    """Median imputation + missing indicators for numbers; one-hot with rare-level grouping for categories.

    ``handle_unknown='infrequent_if_exist'`` maps categories never seen in training
    to the 'infrequent' bucket (or all-zeros), so inference never crashes on new levels.
    """
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=False)),
        ("scale", StandardScaler() if scale_numeric else "passthrough"),
    ])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="__MISSING__")),
        ("onehot", OneHotEncoder(
            handle_unknown="infrequent_if_exist",
            min_frequency=min_frequency,
            max_categories=max_categories,
            sparse_output=False,
        )),
    ])
    return ColumnTransformer(
        [("num", numeric, select_numeric), ("cat", categorical, select_categorical)],
        remainder="drop",
        verbose_feature_names_out=True,
    )


def map_transformed_to_original(transformed_names: Any, numeric_cols: Any, categorical_cols: Any) -> dict[str, str]:
    """Map each transformed column (e.g. ``cat__CODE_X_A``) back to its source column.

    Uses the longest matching source-column prefix within the same transformer
    group, which is robust when one column name is a prefix of another.
    """
    mapping: dict[str, str] = {}
    num_sorted = sorted(map(str, numeric_cols), key=len, reverse=True)
    cat_sorted = sorted(map(str, categorical_cols), key=len, reverse=True)
    for name in map(str, transformed_names):
        group, _, rest = name.partition("__")
        if group == "num":
            base = rest[len("missingindicator_"):] if rest.startswith("missingindicator_") else rest
            mapping[name] = next((c for c in num_sorted if base == c), base)
        elif group == "cat":
            mapping[name] = next((c for c in cat_sorted if rest == c or rest.startswith(c + "_")), rest)
        else:
            mapping[name] = rest or name
    return mapping


@dataclass
class SplitResult:
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray
    strategy: str
    notes: list[str] = field(default_factory=list)
    periods: dict[str, tuple[Any, Any]] = field(default_factory=dict)  # split -> (start, end) for chronological

    def sizes(self) -> dict[str, int]:
        return {"train": len(self.train_idx), "validation": len(self.val_idx), "test": len(self.test_idx)}

    def to_dict(self) -> dict[str, Any]:
        return {"strategy": self.strategy, "sizes": self.sizes(), "notes": self.notes,
                "periods": {k: [str(v[0]), str(v[1])] for k, v in self.periods.items()}}


def _check_classes(y: np.ndarray, idx: np.ndarray, name: str) -> None:
    labels = np.unique(y[idx])
    if len(labels) < 2:
        raise ValueError(
            f"The {name} split contains only one class. Use a larger sample, a different split "
            "proportion, or a random stratified split."
        )


def make_splits(
    y: pd.Series | np.ndarray,
    strategy: str = "stratified",
    val_size: float = 0.2,
    test_size: float = 0.2,
    seed: int = 42,
    dates: pd.Series | None = None,
) -> SplitResult:
    """Positional train/validation/test indices.

    * ``stratified``: random split preserving default prevalence in each part.
    * ``chronological``: earliest records train, later records validate, latest
      records test. Rows without a parseable date are excluded.
    """
    y = np.asarray(y).astype(int)
    n = len(y)
    if n < 30:
        raise ValueError(f"Only {n} labelled rows — at least 30 are needed to create three splits.")
    if not (0 < val_size < 0.5 and 0 < test_size < 0.5 and val_size + test_size < 0.8):
        raise ValueError("Validation and test proportions must each be between 0 and 0.5 and leave at least 20% for training.")
    positions = np.arange(n)
    notes: list[str] = []
    periods: dict[str, tuple[Any, Any]] = {}

    if strategy == "chronological":
        if dates is None:
            raise ValueError("Chronological split requires a date column.")
        d = parse_dates(pd.Series(dates).reset_index(drop=True))
        valid = d.notna().to_numpy()
        if valid.sum() < 30:
            raise ValueError("Too few parseable dates for a chronological split.")
        if (~valid).any():
            notes.append(f"{int((~valid).sum()):,} rows without a valid date were excluded.")
        pos = positions[valid]
        order = np.argsort(d[valid].to_numpy(), kind="stable")
        pos = pos[order]
        n_valid = len(pos)
        n_test = int(round(n_valid * test_size))
        n_val = int(round(n_valid * val_size))
        train_idx = pos[: n_valid - n_val - n_test]
        val_idx = pos[n_valid - n_val - n_test: n_valid - n_test]
        test_idx = pos[n_valid - n_test:]
        for name, idx in (("train", train_idx), ("validation", val_idx), ("test", test_idx)):
            span = d.iloc[idx]
            periods[name] = (pd.Timestamp(span.min()).date(), pd.Timestamp(span.max()).date())
        notes.append("Earliest records train the model, later records select/calibrate it, the latest records test it.")
    elif strategy == "stratified":
        stratify = y if np.min(np.bincount(y)) >= 3 else None
        if stratify is None:
            notes.append("Minority class too small to stratify; a plain random split was used.")
        train_val, test_idx = train_test_split(positions, test_size=test_size, random_state=seed, stratify=stratify)
        rel_val = val_size / (1 - test_size)
        strat2 = y[train_val] if stratify is not None else None
        train_idx, val_idx = train_test_split(train_val, test_size=rel_val, random_state=seed, stratify=strat2)
        notes.append("Random stratified split. It estimates performance on applicants like the training population, "
                     "not necessarily on future applicants (use a chronological split when reliable dates exist).")
    else:
        raise ValueError(f"Unknown split strategy: {strategy}")

    for name, idx in (("training", train_idx), ("validation", val_idx), ("test", test_idx)):
        _check_classes(y, idx, name)
    return SplitResult(np.sort(train_idx), np.sort(val_idx), np.sort(test_idx), strategy, notes, periods)


def stratified_sample(df: pd.DataFrame, y: pd.Series, max_rows: int | None, seed: int = 42) -> tuple[pd.DataFrame, pd.Series]:
    """Down-sample to ``max_rows`` preserving prevalence (keeps original index)."""
    if max_rows is None or len(df) <= max_rows:
        return df, y
    idx, _ = train_test_split(df.index.to_numpy(), train_size=max_rows, random_state=seed, stratify=y.astype(int))
    idx = np.sort(idx)
    return df.loc[idx], y.loc[idx]
