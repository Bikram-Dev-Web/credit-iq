import pandas as pd
import pytest

from src import data_validation as dv


def test_binary_mapping_counts():
    df = pd.DataFrame({"TARGET": [0, 1, 0, 0, 1, 0, 0, 0, 1, 0]})
    m = dv.build_target_mapping(df, "TARGET", [1])
    assert (m.n_positive, m.n_negative, m.n_unmapped) == (3, 7, 0)
    assert m.prevalence == pytest.approx(0.3)


def test_inverted_labels_german_style():
    # UCI German credit: 1 = good, 2 = bad. The user must say which one means default.
    df = pd.DataFrame({"class": [1, 1, 2, 1, 2, 1, 1, 1]})
    m = dv.build_target_mapping(df, "class", [2])
    y = dv.apply_target_mapping(df["class"], m.positive_values, m.negative_values)
    assert y.tolist() == [0, 0, 1, 0, 1, 0, 0, 0]
    assert any("German" in w for w in m.warnings)
    suggested, why = dv.suggest_positive_values(df["class"])
    assert suggested == [] and "German" in why  # never silently assumes 1 = default


def test_text_labels_with_unresolved_loans_are_dropped():
    s = ["Fully Paid", "Charged Off", "Current", "Fully Paid", "Late (31-120 days)", "Charged Off", "Fully Paid"]
    df = pd.DataFrame({"loan_status": s})
    pos, _ = dv.suggest_positive_values(df["loan_status"])
    assert pos == ["Charged Off"]
    neg = dv.suggest_negative_values(df["loan_status"], pos)
    assert neg == ["Fully Paid"]
    m = dv.build_target_mapping(df, "loan_status", pos, neg)
    assert (m.n_positive, m.n_negative, m.n_unmapped) == (2, 3, 2)
    y = dv.apply_target_mapping(df["loan_status"], pos, neg)
    assert y.isna().sum() == 2


def test_missing_target_values_are_unmapped():
    df = pd.DataFrame({"y": [1, 0, None, 0, 1]})
    m = dv.build_target_mapping(df, "y", [1.0])
    assert m.n_unmapped == 1


@pytest.mark.parametrize("kwargs, message", [
    ({"column": "nope", "positive_values": [1]}, "not in the dataset"),
    ({"column": "TARGET", "positive_values": []}, "at least one"),
    ({"column": "TARGET", "positive_values": [5]}, "not found"),
    ({"column": "TARGET", "positive_values": [1], "negative_values": [1, 0]}, "both"),
    ({"column": "TARGET", "positive_values": [0, 1]}, "one class"),
])
def test_invalid_mappings_raise(kwargs, message):
    df = pd.DataFrame({"TARGET": [0, 1, 0, 1, 0]})
    with pytest.raises(dv.TargetMappingError, match=message):
        dv.build_target_mapping(df, **kwargs)


def test_single_class_target_rejected():
    df = pd.DataFrame({"TARGET": [0, 0, 0, 0]})
    with pytest.raises(dv.TargetMappingError):
        dv.build_target_mapping(df, "TARGET", [0])


def test_empty_dataset_rejected():
    with pytest.raises(dv.TargetMappingError, match="empty"):
        dv.build_target_mapping(pd.DataFrame({"TARGET": []}), "TARGET", [1])


def test_high_prevalence_warns_about_inversion():
    df = pd.DataFrame({"TARGET": [1, 1, 1, 0, 1]})
    m = dv.build_target_mapping(df, "TARGET", [1])
    assert any("inverted" in w for w in m.warnings)
