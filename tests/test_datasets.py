"""Tests for drivesense.data.datasets — synthetic images/features only,
no real driver data. Verifies the plumbing (Dataset __len__/__getitem__,
window slicing, subject/group-boundary respect), not model accuracy."""

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from drivesense.data.datasets import (
    DistractionImageDataset,
    FeatureWindowDataset,
    make_feature_windows,
)


def _make_synthetic_images(tmp_path, n=6):
    paths = []
    for i in range(n):
        img = Image.fromarray((np.random.rand(32, 32, 3) * 255).astype("uint8"))
        p = tmp_path / f"img_{i}.png"
        img.save(p)
        paths.append(str(p))
    return paths


def test_distraction_image_dataset_len_and_getitem(tmp_path):
    paths = _make_synthetic_images(tmp_path)
    df = pd.DataFrame({"image_path": paths, "label": [0, 1, 2, 0, 1, 2]})
    ds = DistractionImageDataset(df)
    assert len(ds) == 6
    image, label = ds[0]
    assert image.shape == (3, 32, 32)
    assert image.dtype.is_floating_point
    assert 0.0 <= image.min() and image.max() <= 1.0
    assert label == 0


def test_distraction_image_dataset_requires_expected_columns():
    df = pd.DataFrame({"path": ["a.png"], "y": [0]})
    with pytest.raises(KeyError):
        DistractionImageDataset(df)


def test_distraction_image_dataset_with_custom_transform(tmp_path):
    paths = _make_synthetic_images(tmp_path, n=2)
    df = pd.DataFrame({"image_path": paths, "label": [0, 1]})
    calls = []

    def fake_transform(img):
        calls.append(img.size)
        return np.zeros((3, 8, 8), dtype="float32")

    ds = DistractionImageDataset(df, transform=fake_transform)
    image, label = ds[1]
    assert image.shape == (3, 8, 8)
    assert len(calls) == 1


def _synthetic_feature_df(n_subjects=3, frames_per_subject=20, n_features=3):
    rows = []
    for subj in range(n_subjects):
        label = subj % 2
        for frame in range(frames_per_subject):
            rows.append(
                {
                    "subject_id": f"s{subj}",
                    "frame": frame,
                    "f0": np.random.rand(),
                    "f1": np.random.rand(),
                    "f2": np.random.rand(),
                    "label": label,
                }
            )
    return pd.DataFrame(rows)


def test_make_feature_windows_shapes():
    df = _synthetic_feature_df(n_subjects=3, frames_per_subject=20)
    windows_ds = make_feature_windows(
        df, feature_cols=["f0", "f1", "f2"], label_col="label",
        group_col="subject_id", window_size=5, stride=5,
    )
    assert isinstance(windows_ds, FeatureWindowDataset)
    # 20 frames / window 5 / stride 5 -> 4 windows per subject * 3 subjects
    assert len(windows_ds) == 12
    x, y = windows_ds[0]
    assert x.shape == (5, 3)
    assert y in (0, 1)


def test_make_feature_windows_never_crosses_subject_boundary():
    # Two subjects with distinguishable constant feature values.
    rows = []
    for i in range(10):
        rows.append({"subject_id": "sA", "f0": 1.0, "label": 0})
    for i in range(10):
        rows.append({"subject_id": "sB", "f0": 9.0, "label": 1})
    df = pd.DataFrame(rows)

    windows_ds = make_feature_windows(
        df, feature_cols=["f0"], label_col="label", group_col="subject_id",
        window_size=4, stride=4,
    )
    for i in range(len(windows_ds)):
        x, y = windows_ds[i]
        values = x.numpy().flatten()
        # every value in a window must belong to the same subject (all 1.0 or all 9.0)
        assert np.all(values == values[0])


def test_make_feature_windows_empty_when_group_shorter_than_window():
    df = _synthetic_feature_df(n_subjects=1, frames_per_subject=3)
    windows_ds = make_feature_windows(
        df, feature_cols=["f0", "f1", "f2"], label_col="label",
        group_col="subject_id", window_size=5, stride=5,
    )
    assert len(windows_ds) == 0


def test_make_feature_windows_rejects_invalid_params():
    df = _synthetic_feature_df()
    with pytest.raises(ValueError):
        make_feature_windows(df, ["f0"], "label", "subject_id", window_size=0, stride=1)
    with pytest.raises(ValueError):
        make_feature_windows(df, ["f0"], "label", "subject_id", window_size=5, stride=0)
