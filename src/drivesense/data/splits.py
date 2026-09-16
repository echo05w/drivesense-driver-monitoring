"""Subject-independent train/validation/test splitting.

Rubric requirement (Data & Preprocessing Pipeline, leakage prevention): no
subject's frames/clips may appear in more than one split. This module splits
by a subject/group ID column, never by row, using sklearn's GroupShuffleSplit.
"""

from __future__ import annotations

from typing import Tuple

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


def subject_independent_split(
    df: pd.DataFrame,
    subject_col: str,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split a dataframe into train/val/test with no subject overlap.

    Args:
        df: one row per sample (image/clip), with a subject-identifying column.
        subject_col: column name holding the subject/driver ID.
        test_size: fraction of *subjects* (not rows) held out for the test set.
        val_size: fraction of the *remaining* subjects held out for validation.
        random_state: seed for reproducibility.

    Returns:
        (train_df, val_df, test_df), each a row subset of `df`.
    """
    if subject_col not in df.columns:
        raise KeyError(f"subject_col '{subject_col}' not found in dataframe columns")
    if not 0 < test_size < 1:
        raise ValueError("test_size must be in (0, 1)")
    if not 0 <= val_size < 1:
        raise ValueError("val_size must be in [0, 1)")

    groups = df[subject_col]

    gss_test = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    trainval_idx, test_idx = next(gss_test.split(df, groups=groups))
    trainval_df = df.iloc[trainval_idx]
    test_df = df.iloc[test_idx]

    if val_size == 0:
        return trainval_df, trainval_df.iloc[0:0], test_df

    gss_val = GroupShuffleSplit(n_splits=1, test_size=val_size, random_state=random_state)
    train_idx, val_idx = next(
        gss_val.split(trainval_df, groups=trainval_df[subject_col])
    )
    train_df = trainval_df.iloc[train_idx]
    val_df = trainval_df.iloc[val_idx]

    assert_no_subject_leakage(train_df, val_df, test_df, subject_col)
    return train_df, val_df, test_df


def assert_no_subject_leakage(
    train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, subject_col: str
) -> None:
    """Raise AssertionError if any subject appears in more than one split."""
    train_subjects = set(train_df[subject_col])
    val_subjects = set(val_df[subject_col])
    test_subjects = set(test_df[subject_col])

    overlap_tv = train_subjects & val_subjects
    overlap_tt = train_subjects & test_subjects
    overlap_vt = val_subjects & test_subjects

    if overlap_tv or overlap_tt or overlap_vt:
        raise AssertionError(
            "Subject leakage detected across splits: "
            f"train/val={overlap_tv}, train/test={overlap_tt}, val/test={overlap_vt}"
        )
