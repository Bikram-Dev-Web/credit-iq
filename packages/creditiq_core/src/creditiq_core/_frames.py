"""Small typing helpers for pandas."""
from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pandas as pd


def iter_rows(df: pd.DataFrame, index: bool = True) -> Iterator[Any]:
    """``DataFrame.itertuples`` with rows typed as ``Any``.

    pandas-stubs types every field of an itertuples row as the union of all scalar types,
    which makes ordinary attribute access (``r.column``, ``r.unique``) a type error.
    """
    return iter(df.itertuples(index=index))
