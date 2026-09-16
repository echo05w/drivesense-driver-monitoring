"""Geometric signals derived from facial landmarks (e.g., MediaPipe Face Mesh).

These are pure functions of landmark coordinates so they can be unit-tested
without any model or dataset dependency. Landmark indices follow the
MediaPipe Face Mesh topology and must be mapped to the correct point
indices when wired up to a real landmarker in `drivesense.features.landmarks`.
"""

from __future__ import annotations

import math
from typing import Sequence, Tuple

Point = Tuple[float, float]


def _dist(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def eye_aspect_ratio(eye_points: Sequence[Point]) -> float:
    """Compute the Eye Aspect Ratio (EAR) from 6 eye landmark points.

    Expects points ordered as [p1, p2, p3, p4, p5, p6] following the
    classic EAR formulation (Soukupova & Cech, 2016):
        p1, p4 = horizontal corners
        p2, p3 = upper eyelid points
        p5, p6 = lower eyelid points

    EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)

    A low EAR indicates a closed or closing eye.
    """
    if len(eye_points) != 6:
        raise ValueError(f"eye_aspect_ratio expects exactly 6 points, got {len(eye_points)}")
    p1, p2, p3, p4, p5, p6 = eye_points
    horizontal = _dist(p1, p4)
    if horizontal == 0:
        raise ValueError("Degenerate eye landmarks: horizontal distance is zero")
    vertical = _dist(p2, p6) + _dist(p3, p5)
    return vertical / (2.0 * horizontal)


def mouth_aspect_ratio(mouth_points: Sequence[Point]) -> float:
    """Compute the Mouth Aspect Ratio (MAR) from 6 mouth landmark points.

    Same structural formulation as EAR but for the outer mouth contour.
    A high MAR indicates an open mouth (e.g., yawning).
    """
    if len(mouth_points) != 6:
        raise ValueError(f"mouth_aspect_ratio expects exactly 6 points, got {len(mouth_points)}")
    p1, p2, p3, p4, p5, p6 = mouth_points
    horizontal = _dist(p1, p4)
    if horizontal == 0:
        raise ValueError("Degenerate mouth landmarks: horizontal distance is zero")
    vertical = _dist(p2, p6) + _dist(p3, p5)
    return vertical / (2.0 * horizontal)
