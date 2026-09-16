"""PyTorch Dataset classes and sequence-windowing utilities.

These operate on already-extracted metadata (a dataframe of image paths +
labels, or a dataframe of per-frame landmark features) — they do not
themselves download or extract anything, so they are fully testable with
synthetic data without needing the real datasets or a MediaPipe graph.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


class DistractionImageDataset(Dataset):
    """Single-frame image dataset for the distraction classification task.

    Expects a dataframe with at least columns [image_path, label] (label as
    an integer class index — encode class names to indices before
    constructing this, e.g. with sklearn.preprocessing.LabelEncoder, and
    keep the encoder alongside the model artifact for inference).
    """

    def __init__(
        self,
        df: pd.DataFrame,
        image_path_col: str = "image_path",
        label_col: str = "label",
        transform: Optional[Callable] = None,
    ) -> None:
        if image_path_col not in df.columns or label_col not in df.columns:
            raise KeyError(
                f"DataFrame must contain '{image_path_col}' and '{label_col}' columns"
            )
        self.df = df.reset_index(drop=True)
        self.image_path_col = image_path_col
        self.label_col = label_col
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        image_path = row[self.image_path_col]
        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        else:
            image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
        label = int(row[self.label_col])
        return image, label


class FeatureWindowDataset(Dataset):
    """Windowed landmark-feature sequences for the drowsiness temporal models.

    Takes a per-frame feature dataframe and slices it into fixed-length,
    fixed-stride windows *within* each subject/clip group — a window never
    spans two different subjects/clips, which would otherwise silently mix
    unrelated temporal signals.
    """

    def __init__(
        self,
        windows: np.ndarray,
        labels: np.ndarray,
    ) -> None:
        if len(windows) != len(labels):
            raise ValueError("windows and labels must have the same length")
        self.windows = windows
        self.labels = labels

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int):
        return (
            torch.tensor(self.windows[idx], dtype=torch.float32),
            int(self.labels[idx]),
        )


def make_feature_windows(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    label_col: str,
    group_col: str,
    window_size: int,
    stride: int,
) -> "FeatureWindowDataset":
    """Slice a per-frame feature dataframe into fixed-length windows per group.

    Args:
        df: one row per frame, sorted by time within each group (caller's
            responsibility — this function does not re-sort, to avoid
            silently hiding an upstream ordering bug).
        feature_cols: columns to include per frame (e.g. EAR/MAR/head-pose).
        label_col: the label for the *window* (assumed constant within a
            group — e.g. one drowsiness-state label per clip).
        group_col: subject/clip identifier; windows never cross group
            boundaries.
        window_size: number of frames per window.
        stride: step between consecutive window start indices.

    Returns:
        A FeatureWindowDataset ready for use with a DataLoader.
    """
    if window_size <= 0 or stride <= 0:
        raise ValueError("window_size and stride must be positive")

    windows: List[np.ndarray] = []
    labels: List = []

    for _, group_df in df.groupby(group_col, sort=False):
        values = group_df[list(feature_cols)].to_numpy(dtype=np.float32)
        group_label = group_df[label_col].iloc[0]
        n_frames = len(values)
        for start in range(0, max(n_frames - window_size + 1, 0), stride):
            windows.append(values[start : start + window_size])
            labels.append(group_label)

    if not windows:
        return FeatureWindowDataset(
            windows=np.empty((0, window_size, len(feature_cols)), dtype=np.float32),
            labels=np.empty((0,)),
        )

    return FeatureWindowDataset(windows=np.stack(windows), labels=np.array(labels))
