import io

import numpy as np
import pandas as pd
import pytest

from src import data_loader as dl
from src import data_validation as dv


def _toy():
    return pd.DataFrame({
        "customer_id": np.arange(100),
        "income": np.r_[np.linspace(1, 99, 98), np.inf, np.nan],
        "empty": [np.nan] * 100,
        "const": [1] * 100,
        "gender": ["F", "M"] * 50,
        "total_rec_prncp": np.arange(100) * 2.0,
        "app_date": pd.date_range("2024-01-01", periods=100).strftime("%Y-%m-%d"),
        "grade": list("ABCD") * 25,
        "y": [0, 1] * 50,
    })


def test_profile_counts_and_infinite_values():
    summary, prof = dv.profile_dataframe(_toy())
    assert summary["rows"] == 100 and summary["columns"] == 9
    income = prof.set_index("column").loc["income"]
    assert income["n_inf"] == 1 and income["missing"] == 1
    assert prof.set_index("column").loc["empty", "kind"] == dv.KIND_EMPTY


def test_issues_flag_quality_problems():
    issues = dv.find_issues(_toy())
    cats = {(i.column, i.category) for i in issues}
    assert ("empty", "all_missing") in cats
    assert ("income", "infinite") in cats
    assert ("const", "constant") in cats
    assert ("—", "small_data") in cats


def test_empty_dataset():
    with pytest.raises(ValueError):
        dv.profile_dataframe(pd.DataFrame())
    issues = dv.find_issues(pd.DataFrame({"a": []}))
    assert issues[0].severity == "error"


def test_suggested_roles():
    roles = dv.suggest_feature_roles(_toy(), "y", "app_date").set_index("column")
    assert roles.loc["customer_id", "role"] == "identifier" and not roles.loc["customer_id", "use"]
    assert roles.loc["empty", "role"] == "empty"
    assert roles.loc["const", "role"] == "constant"
    assert roles.loc["gender", "role"] == "sensitive" and not roles.loc["gender", "use"]
    assert roles.loc["total_rec_prncp", "role"] == "leakage"
    assert roles.loc["app_date", "role"] == "date"
    assert roles.loc["y", "role"] == "target"
    assert roles.loc["grade", "use"] and roles.loc["income", "use"]


def test_date_detection_and_missing_dates():
    df = _toy()
    assert dv.detect_date_columns(df) == ["app_date"]
    assert dv.detect_date_columns(df.drop(columns="app_date")) == []


def test_leakage_scan_flags_post_outcome_column(synthetic_df):
    y = dv.apply_target_mapping(synthetic_df["TARGET"], [1], [0])
    scan = dv.leakage_scan(synthetic_df, y, ["POST_LOAN_COLLECTION_CALLS", "EXT_SOURCE_2", "NAME_CONTRACT_TYPE"]).set_index("column")
    assert scan.loc["POST_LOAN_COLLECTION_CALLS", "assessment"].startswith("SUSPICIOUS")
    assert not scan.loc["EXT_SOURCE_2", "assessment"].startswith("SUSPICIOUS")
    assert 0.5 <= scan.loc["NAME_CONTRACT_TYPE", "single_feature_auc"] < 0.9


def test_synthetic_roles_exclude_leaky_and_sensitive(synthetic_df):
    roles = dv.suggest_feature_roles(synthetic_df, "TARGET", "APPLICATION_DATE").set_index("column")
    for col in ("SK_ID_CURR", "POST_LOAN_COLLECTION_CALLS", "CODE_GENDER", "DAYS_BIRTH", "APPLICATION_DATE"):
        assert not roles.loc[col, "use"], col


def test_sanitize_filename():
    assert dl.sanitize_filename("../../etc/passwd") == "passwd"
    assert dl.sanitize_filename("my report (final).md") == "my_report_final_.md"
    assert dl.sanitize_filename("...") == "file"


def test_read_csv_from_bytes_and_sampling():
    df = pd.DataFrame({"a": range(1000), "b": ["x"] * 1000})
    raw = df.to_csv(index=False).encode()
    out, info = dl.read_csv_file(raw, name="t.csv", sample_rows=100)
    assert len(out) == 100 and info.sampled and info.n_rows_file == 1000
    _, info = dl.read_csv_file(io.BytesIO(b"a,a\n1,2\n"), name="dup.csv")
    assert any("Duplicate column headers" in n for n in info.notes)
    with pytest.raises(Exception):
        dl.read_csv_file(io.BytesIO(b""), name="empty.csv")


def test_read_local_csv_chunked_sampling(tmp_path):
    p = tmp_path / "big.csv"
    pd.DataFrame({"a": range(5000)}).to_csv(p, index=False)
    out, info = dl.read_csv_file(p, sample_rows=500, source_kind="local")
    assert info.sampled and 300 < len(out) < 700  # chunk-wise Bernoulli sampling ≈ 500 rows


def test_synthetic_generator_is_reproducible_and_labelled():
    a = dl.generate_synthetic_credit_data(500, seed=3)
    b = dl.generate_synthetic_credit_data(500, seed=3)
    pd.testing.assert_frame_equal(a, b)
    assert 0.03 < a["TARGET"].mean() < 0.15
    df, info = dl.load_synthetic_dataset(1_000, seed=3)
    assert info.is_synthetic and any("SYNTHETIC" in n for n in info.notes)


def test_quality_report_contains_only_computed_content():
    df = _toy()
    summary, prof = dv.profile_dataframe(df)
    md = dv.quality_report({"name": "toy", "is_synthetic": False}, summary, prof, dv.find_issues(df, prof), None, None)
    assert "Data Quality Report" in md and "income" in md and "TODO" not in md
