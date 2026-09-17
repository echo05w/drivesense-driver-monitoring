"""State Farm Distracted Driver Detection: metadata loading and validation.

This module turns the competition's raw `driver_imgs_list.csv` + extracted
`imgs/train/<classname>/<img>.jpg` layout into a plain dataframe the generic
`DistractionImageDataset` (see `datasets.py`) and `subject_independent_split`
(see `splits.py`) already know how to consume. It does not guess the on-disk
layout silently: `load_metadata` fails loudly if an expected image is
missing rather than skipping it, so a wrong assumption about the extracted
structure is caught immediately instead of silently shrinking the dataset.

CLASS_NAMES follows the competition's official mapping (c0-c9) — see
https://www.kaggle.com/competitions/state-farm-distracted-driver-detection/data
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from PIL import Image, UnidentifiedImageError

CLASS_NAMES: Dict[str, str] = {
    "c0": "safe_driving",
    "c1": "texting_right",
    "c2": "talking_phone_right",
    "c3": "texting_left",
    "c4": "talking_phone_left",
    "c5": "operating_radio",
    "c6": "drinking",
    "c7": "reaching_behind",
    "c8": "hair_and_makeup",
    "c9": "talking_to_passenger",
}

# Classes c1-c4 are direction-specific (left vs. right hand/phone use).
# Horizontal flipping would silently swap these labels, so it must never be
# used as an augmentation for this dataset — documented here so it isn't
# reintroduced by a future session copying a generic image-augmentation recipe.
HORIZONTAL_FLIP_SAFE = False

CLASS_ORDER: List[str] = sorted(CLASS_NAMES.keys())  # c0..c9 -> label indices 0..9


def load_metadata(raw_dir: Path) -> pd.DataFrame:
    """Build a (image_path, label, subject, classname) dataframe from the
    extracted competition data.

    Args:
        raw_dir: directory containing `driver_imgs_list.csv` and an
            `imgs/train/<classname>/<img>` tree (the extracted archive root).

    Raises:
        FileNotFoundError: if the CSV or any listed image is missing — this
            is intentional; a partially-extracted or wrongly-laid-out archive
            must fail loudly, not silently produce a smaller dataset.
    """
    raw_dir = Path(raw_dir)
    csv_path = raw_dir / "driver_imgs_list.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"driver_imgs_list.csv not found at {csv_path}")

    df = pd.read_csv(csv_path)
    expected_cols = {"subject", "classname", "img"}
    if not expected_cols.issubset(df.columns):
        raise ValueError(f"Unexpected CSV columns {list(df.columns)}, expected {expected_cols}")

    unknown_classes = set(df["classname"]) - set(CLASS_NAMES)
    if unknown_classes:
        raise ValueError(f"Unknown class codes in CSV not in CLASS_NAMES: {unknown_classes}")

    image_root = raw_dir / "imgs" / "train"
    df = df.copy()
    df["image_path"] = df.apply(
        lambda row: str(image_root / row["classname"] / row["img"]), axis=1
    )

    missing = [p for p in df["image_path"] if not Path(p).exists()]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} images listed in driver_imgs_list.csv were not found on disk "
            f"(expected under {image_root}/<classname>/<img>). First few: {missing[:5]}. "
            "This means the archive layout does not match the assumed "
            "imgs/train/<classname>/<img> structure - re-check the extraction."
        )

    df["label"] = df["classname"].map({c: i for i, c in enumerate(CLASS_ORDER)})
    return df[["image_path", "label", "classname", "subject", "img"]].reset_index(drop=True)


@dataclass
class ImageIntegrityReport:
    total_checked: int
    corrupt_paths: List[str]
    dimensions: List[Tuple[int, int]]  # (width, height) for every successfully-opened image

    @property
    def n_corrupt(self) -> int:
        return len(self.corrupt_paths)


def validate_images(image_paths: List[str]) -> ImageIntegrityReport:
    """Actually open every image and record its real dimensions.

    Not a sampled check - the point of this function is to catch corrupt or
    truncated files before they cause a training-time crash hours into a run.
    For ~22k small JPEGs this is a few seconds of I/O, not a bottleneck.
    """
    corrupt: List[str] = []
    dims: List[Tuple[int, int]] = []
    for p in image_paths:
        try:
            with Image.open(p) as img:
                img.verify()
            with Image.open(p) as img:  # re-open: verify() leaves the file unusable for further ops
                dims.append(img.size)
        except (UnidentifiedImageError, OSError) as exc:
            corrupt.append(f"{p} ({exc})")
    return ImageIntegrityReport(total_checked=len(image_paths), corrupt_paths=corrupt, dimensions=dims)
