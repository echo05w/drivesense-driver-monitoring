"""Unit tests for drivesense.data.splits — verifies subject-independent
splitting on a synthetic dataframe (not a real dataset)."""

import pandas as pd
import pytest

from drivesense.data.splits import assert_no_subject_leakage, subject_independent_split


def _make_synthetic_df(n_subjects: int = 20, rows_per_subject: int = 10) -> pd.DataFrame:
    rows = []
    for subject_id in range(n_subjects):
        for _ in range(rows_per_subject):
            rows.append({"subject_id": f"s{subject_id}", "label": subject_id % 3})
    return pd.DataFrame(rows)


def test_split_has_no_subject_overlap():
    df = _make_synthetic_df()
    train_df, val_df, test_df = subject_independent_split(
        df, subject_col="subject_id", test_size=0.2, val_size=0.2, random_state=0
    )
    assert_no_subject_leakage(train_df, val_df, test_df, "subject_id")


def test_split_covers_all_rows_exactly_once():
    df = _make_synthetic_df()
    train_df, val_df, test_df = subject_independent_split(
        df, subject_col="subject_id", test_size=0.2, val_size=0.2, random_state=0
    )
    total = len(train_df) + len(val_df) + len(test_df)
    assert total == len(df)


def test_missing_subject_column_raises():
    df = _make_synthetic_df()
    with pytest.raises(KeyError):
        subject_independent_split(df, subject_col="does_not_exist")


def test_invalid_test_size_raises():
    df = _make_synthetic_df()
    with pytest.raises(ValueError):
        subject_independent_split(df, subject_col="subject_id", test_size=1.5)
