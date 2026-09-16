"""Unit tests for drivesense.utils.geometry.

Uses synthetic, hand-constructed coordinates (not real facial landmarks) to
verify the EAR/MAR math itself is correct. These are NOT project results —
see docs/Rubric_Alignment.md's academic-integrity note.
"""

import math

import pytest

from drivesense.utils.geometry import eye_aspect_ratio, mouth_aspect_ratio


def test_ear_wide_open_eye_is_high():
    # A rectangle-ish eye shape: wide horizontally, tall vertically.
    p1, p4 = (0.0, 0.0), (10.0, 0.0)  # corners
    p2, p3 = (3.0, 3.0), (7.0, 3.0)  # upper lid
    p5, p6 = (7.0, -3.0), (3.0, -3.0)  # lower lid
    ear = eye_aspect_ratio([p1, p2, p3, p4, p5, p6])
    assert ear == pytest.approx(0.6, rel=1e-6)


def test_ear_closed_eye_is_near_zero():
    p1, p4 = (0.0, 0.0), (10.0, 0.0)
    p2, p3 = (3.0, 0.01), (7.0, 0.01)
    p5, p6 = (7.0, -0.01), (3.0, -0.01)
    ear = eye_aspect_ratio([p1, p2, p3, p4, p5, p6])
    assert ear < 0.05


def test_ear_rejects_wrong_point_count():
    with pytest.raises(ValueError):
        eye_aspect_ratio([(0, 0), (1, 1)])


def test_ear_rejects_degenerate_horizontal():
    p1 = p4 = (5.0, 5.0)
    p2, p3, p5, p6 = (0, 0), (0, 0), (0, 0), (0, 0)
    with pytest.raises(ValueError):
        eye_aspect_ratio([p1, p2, p3, p4, p5, p6])


def test_mar_open_mouth_is_high():
    p1, p4 = (0.0, 0.0), (10.0, 0.0)
    p2, p3 = (3.0, 5.0), (7.0, 5.0)
    p5, p6 = (7.0, -5.0), (3.0, -5.0)
    mar = mouth_aspect_ratio([p1, p2, p3, p4, p5, p6])
    assert mar == pytest.approx(1.0, rel=1e-6)


def test_mar_rejects_wrong_point_count():
    with pytest.raises(ValueError):
        mouth_aspect_ratio([(0, 0)] * 5)
