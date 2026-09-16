"""Face-landmark feature extraction (MediaPipe Face Mesh wrapper).

STATUS: not yet implemented / not yet runnable in this environment —
`mediapipe` is not installed locally as of this commit (see
docs/Master_Plan_Status.md environment notes). This module defines the
intended interface so downstream code (drowsiness feature pipeline) can be
written and tested against it with a fake/stub extractor before the real
dependency is installed (locally or in Colab).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Protocol, Tuple

Point = Tuple[float, float]

# MediaPipe Face Mesh landmark indices for the 6-point EAR/MAR formulation.
# TODO(verify): confirm exact indices against the installed MediaPipe version
# before relying on them for real feature extraction — these are the commonly
# cited indices for the 468-point face mesh and must be validated, not assumed.
LEFT_EYE_IDX = (33, 160, 158, 133, 153, 144)
RIGHT_EYE_IDX = (362, 385, 387, 263, 373, 380)
OUTER_MOUTH_IDX = (61, 39, 269, 291, 405, 181)


@dataclass
class FrameLandmarkFeatures:
    left_ear: float
    right_ear: float
    mar: float
    head_pitch_deg: float
    head_yaw_deg: float
    head_roll_deg: float
    face_detected: bool


class LandmarkExtractor(Protocol):
    """Interface a real (MediaPipe-backed) or fake (test) extractor must satisfy."""

    def extract(self, frame_bgr) -> FrameLandmarkFeatures: ...


class MediaPipeLandmarkExtractor:
    """Real extractor. Requires `mediapipe` and `opencv-python` installed.

    Not yet exercised by any test in this repository — construction will
    raise ImportError until the dependency is installed, which is expected
    and intentional (fails loudly rather than silently no-op-ing).
    """

    def __init__(self) -> None:
        try:
            import mediapipe as mp  # noqa: F401
        except ImportError as exc:  # pragma: no cover - environment-dependent
            raise ImportError(
                "mediapipe is required for MediaPipeLandmarkExtractor; "
                "install it locally or run this in a Colab runtime with "
                "requirements.txt installed."
            ) from exc
        raise NotImplementedError(
            "MediaPipeLandmarkExtractor pipeline is not yet implemented — "
            "tracked in docs/Master_Plan_Status.md, phase 5 (preprocessing)."
        )

    def extract(self, frame_bgr) -> FrameLandmarkFeatures:  # pragma: no cover
        raise NotImplementedError
