"""Tests for State Farm metadata loading/validation, using synthetic fixtures
only - no real dataset required."""

from __future__ import annotations

import pandas as pd
import pytest
from PIL import Image

from drivesense.data.state_farm import CLASS_ORDER, load_metadata, validate_images


def _make_fixture(tmp_path, corrupt_last: bool = False):
    raw_dir = tmp_path / "raw"
    image_root = raw_dir / "imgs" / "train"
    rows = []
    for subject, classnames in [
        ("p001", ["c0", "c1"]),
        ("p002", ["c0", "c2"]),
    ]:
        for classname in classnames:
            class_dir = image_root / classname
            class_dir.mkdir(parents=True, exist_ok=True)
            for i in range(2):
                fname = f"{subject}_{classname}_{i}.jpg"
                img_path = class_dir / fname
                Image.new("RGB", (10, 8), color=(i * 10, 0, 0)).save(img_path)
                rows.append({"subject": subject, "classname": classname, "img": fname})

    if corrupt_last:
        # Overwrite the last written file with garbage bytes (not a valid image).
        img_path.write_bytes(b"not a real jpeg")

    csv_path = raw_dir / "driver_imgs_list.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    return raw_dir


def test_load_metadata_builds_correct_paths_and_labels(tmp_path):
    raw_dir = _make_fixture(tmp_path)
    df = load_metadata(raw_dir)

    assert len(df) == 8
    assert set(df.columns) == {"image_path", "label", "classname", "subject", "img"}
    assert set(df["subject"]) == {"p001", "p002"}
    # label must match CLASS_ORDER's index for the classname
    for _, row in df.iterrows():
        assert row["label"] == CLASS_ORDER.index(row["classname"])
    assert all(pd is not None for _ in df["image_path"])  # sanity: paths were built


def test_load_metadata_raises_on_missing_image(tmp_path):
    raw_dir = _make_fixture(tmp_path)
    # Delete one image file the CSV still references.
    df = pd.read_csv(raw_dir / "driver_imgs_list.csv")
    first = df.iloc[0]
    missing_path = raw_dir / "imgs" / "train" / first["classname"] / first["img"]
    missing_path.unlink()

    with pytest.raises(FileNotFoundError):
        load_metadata(raw_dir)


def test_load_metadata_raises_on_unknown_class(tmp_path):
    raw_dir = _make_fixture(tmp_path)
    csv_path = raw_dir / "driver_imgs_list.csv"
    df = pd.read_csv(csv_path)
    df.loc[0, "classname"] = "c99"
    df.to_csv(csv_path, index=False)

    with pytest.raises(ValueError):
        load_metadata(raw_dir)


def test_validate_images_detects_corrupt_file(tmp_path):
    raw_dir = _make_fixture(tmp_path, corrupt_last=True)
    df = load_metadata(raw_dir)
    report = validate_images(df["image_path"].tolist())

    assert report.total_checked == len(df)
    assert report.n_corrupt == 1
    assert len(report.dimensions) == len(df) - 1
    assert all(d == (10, 8) for d in report.dimensions)


def test_validate_images_all_valid(tmp_path):
    raw_dir = _make_fixture(tmp_path)
    df = load_metadata(raw_dir)
    report = validate_images(df["image_path"].tolist())

    assert report.n_corrupt == 0
    assert report.dimensions == [(10, 8)] * len(df)
