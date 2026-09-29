"""Lightweight tests for drivesense.demo's deterministic, non-model-loading logic.

Full pipeline (model loading + inference on real frames) is exercised via
manual smoke tests (webcam, a real UTA-RLDD video, a real State Farm image -
see README) rather than the automated suite, since it needs real checkpoint
files / real video devices that may not exist in every environment (CI).
"""

import pytest

from drivesense.demo import open_source


def test_open_source_invalid_webcam_index_raises_clear_error():
    with pytest.raises(RuntimeError, match="webcam"):
        open_source("97")  # index very unlikely to exist


def test_open_source_missing_video_file_raises():
    with pytest.raises(RuntimeError):
        open_source("data/raw/does_not_exist.mp4")


def test_open_source_reads_real_image(tmp_path):
    import cv2
    import numpy as np

    img_path = tmp_path / "frame.jpg"
    cv2.imwrite(str(img_path), np.zeros((10, 10, 3), dtype="uint8"))
    kind, frame = open_source(str(img_path))
    assert kind == "image"
    assert frame.shape == (10, 10, 3)
