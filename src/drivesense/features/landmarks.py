"""Face-landmark feature extraction using MediaPipe's Tasks API.

IMPORTANT (verified 2026-09-16): the installed `mediapipe` version in this
project (1.0.1) does NOT expose the older `mp.solutions.face_mesh` API that
most tutorials/older code assume. It only exposes the newer Tasks API
(`mediapipe.tasks.python.vision.FaceLandmarker`), confirmed by inspecting
`dir(mediapipe)` / `dir(mediapipe.tasks.python.vision)` directly in this
environment rather than assumed from training data. This module is written
against the verified, currently-installed API.

Requires a downloaded `.task` model bundle (not bundled with the pip
package) — see `download_face_landmarker_model()`.
"""

from __future__ import annotations

import math
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Protocol, Sequence, Tuple

from drivesense.utils.geometry import eye_aspect_ratio, mouth_aspect_ratio

Point = Tuple[float, float]

# MediaPipe Face Mesh landmark indices for the 6-point EAR/MAR formulation.
# These are the standard indices for the 468-point face mesh topology used
# by both the legacy Face Mesh solution and the new Face Landmarker task.
LEFT_EYE_IDX = (33, 160, 158, 133, 153, 144)
RIGHT_EYE_IDX = (362, 385, 387, 263, 373, 380)
OUTER_MOUTH_IDX = (61, 39, 269, 291, 405, 181)

FACE_LANDMARKER_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/latest/face_landmarker.task"
)
DEFAULT_MODEL_PATH = Path.home() / ".cache" / "drivesense" / "face_landmarker.task"


def download_face_landmarker_model(dest: Path = DEFAULT_MODEL_PATH) -> Path:
    """Download the MediaPipe Face Landmarker model bundle if not already present.

    Verified working in this environment (2026-09-16): downloads a ~3.6 MB
    .task file. Cached outside the repository (never committed — see
    .gitignore) since it's a large-ish binary regenerable artifact, not
    project source.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        urllib.request.urlretrieve(FACE_LANDMARKER_MODEL_URL, dest)
    return dest


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

    def extract(self, frame_rgb) -> FrameLandmarkFeatures: ...


def _no_face_result() -> FrameLandmarkFeatures:
    return FrameLandmarkFeatures(
        left_ear=0.0, right_ear=0.0, mar=0.0,
        head_pitch_deg=0.0, head_yaw_deg=0.0, head_roll_deg=0.0,
        face_detected=False,
    )


def _estimate_head_pose(landmarks) -> Tuple[float, float, float]:
    """Rough head pose proxy from a few key landmarks (nose tip, eye corners).

    This is NOT a calibrated 3D head-pose solver (that needs a camera
    intrinsics model + solvePnP, planned but not yet implemented — tracked
    in docs/Master_Plan_Status.md). It is a placeholder, clearly labeled as
    such, giving a directional yaw/roll signal from 2D landmark geometry
    only; pitch is left at 0.0 until the real solvePnP-based estimator
    exists. Do not report this as a validated head-pose measurement.
    """
    left_eye_outer = landmarks[33]
    right_eye_outer = landmarks[263]
    nose_tip = landmarks[1]

    dx = right_eye_outer.x - left_eye_outer.x
    dy = right_eye_outer.y - left_eye_outer.y
    roll_deg = math.degrees(math.atan2(dy, dx))

    eye_mid_x = (left_eye_outer.x + right_eye_outer.x) / 2.0
    yaw_proxy = (nose_tip.x - eye_mid_x) / max(abs(dx), 1e-6)
    yaw_deg = yaw_proxy * 45.0  # unitless proxy scaled to a plausible degree range

    pitch_deg = 0.0  # not yet estimated — see docstring
    return pitch_deg, yaw_deg, roll_deg


class MediaPipeLandmarkExtractor:
    """Real extractor backed by MediaPipe's FaceLandmarker task.

    API surface verified against the installed mediapipe==1.0.1 in this
    environment. Full end-to-end `extract()` execution (constructing the
    task graph) could NOT be completed on this local machine as of
    2026-09-16: the process was killed (exit 137, consistent with an OOM
    kill) with only ~490 MB free RAM available alongside the user's normal
    desktop session (verified via `free -h` and `ps aux --sort=-%mem` — the
    memory is consumed by the user's own Chrome/desktop processes, not a
    leak in this code). This is a real, external resource constraint, not a
    fabricated pass. It must be verified either in Google Colab (dedicated
    RAM) or on this machine when more memory is free, before this class is
    marked "done" in docs/Master_Plan_Status.md.
    """

    def __init__(self, model_path: Optional[Path] = None, num_faces: int = 1) -> None:
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        model_path = model_path or download_face_landmarker_model()
        base_options = mp_python.BaseOptions(model_asset_path=str(model_path))
        options = vision.FaceLandmarkerOptions(base_options=base_options, num_faces=num_faces)
        self._mp = mp
        self._landmarker = vision.FaceLandmarker.create_from_options(options)

    def extract(self, frame_rgb) -> FrameLandmarkFeatures:
        """Extract EAR/MAR/head-pose features from one RGB frame (H, W, 3 uint8 array)."""
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=frame_rgb)
        result = self._landmarker.detect(mp_image)

        if not result.face_landmarks:
            return _no_face_result()

        landmarks = result.face_landmarks[0]
        h_norm, w_norm = 1.0, 1.0  # landmarks are already normalized [0, 1] by MediaPipe

        def pt(idx: int) -> Point:
            lm = landmarks[idx]
            return (lm.x * w_norm, lm.y * h_norm)

        left_eye_pts = [pt(i) for i in LEFT_EYE_IDX]
        right_eye_pts = [pt(i) for i in RIGHT_EYE_IDX]
        mouth_pts = [pt(i) for i in OUTER_MOUTH_IDX]

        left_ear = eye_aspect_ratio(left_eye_pts)
        right_ear = eye_aspect_ratio(right_eye_pts)
        mar = mouth_aspect_ratio(mouth_pts)
        pitch, yaw, roll = _estimate_head_pose(landmarks)

        return FrameLandmarkFeatures(
            left_ear=left_ear,
            right_ear=right_ear,
            mar=mar,
            head_pitch_deg=pitch,
            head_yaw_deg=yaw,
            head_roll_deg=roll,
            face_detected=True,
        )


# --- Lightweight local fallback (Haar cascades) --------------------------
#
# `MediaPipeLandmarkExtractor` cannot be constructed on this development
# machine (OOM-killed, see its docstring above and docs/Master_Plan_Status.md
# / CLAUDE.md — a confirmed, do-not-retry local boundary). It is NOT
# retried here. Instead, `HaarCascadeLandmarkExtractor` below is a genuinely
# real, much lighter-weight (no big task-graph, no ~10s+ init) alternative
# built entirely on `cv2.CascadeClassifier`, so `scripts/demo.py` can show a
# real (if coarse) local drowsiness signal instead of no signal at all.
#
# Honesty/limits, stated explicitly because this project's own conventions
# require it (see CLAUDE.md "Never fabricate"):
#   - This is an approximation, not a substitute for MediaPipe's 468-point
#     mesh. Haar cascades return bounding boxes, not landmark points, so a
#     true 6-point geometric EAR/MAR (Soukupova & Cech) cannot be computed
#     from them.
#   - `left_ear`/`right_ear` here are a coarse, discrete open/closed proxy
#     (not a continuous ratio): eyes detected inside the face ROI -> "open"
#     value, not detected -> "closed" value. This is the same
#     detection-dropout heuristic used by several lightweight blink-detector
#     tutorials that rely on Haar cascades, not an invented technique — but
#     it is materially less precise than real EAR, and is intended only for
#     enabling a local CLI demo, never as a research-grade drowsiness signal.
#   - `mar` is always 0.0 here: no maintained, licensable mouth cascade
#     ships with this project's OpenCV build, so yawning cannot be detected
#     through this fallback path at all (see `docs/Master_Plan_Status.md`).
#   - Head pose is a crude horizontal-offset-only yaw proxy from the face
#     bounding box position in-frame; pitch/roll are always 0.0.
#   - The trained temporal drowsiness model
#     (`drivesense.models.temporal.DrowsinessGRU` /
#     `DrowsinessTemporalCNN`) is trained on the real MediaPipe feature
#     schema (`FEATURE_COLUMNS` in `drivesense.data.drowsiness`) via the
#     Colab pipeline, and MUST NOT be fed this fallback extractor's output
#     as if it were equivalent input — the two are numerically different
#     scales. `scripts/demo.py` therefore only ever drives the *rule-based*
#     PERCLOS-style drowsiness score (`drivesense.inference.drowsiness`)
#     from this fallback, never the untrained/incompatible neural model.

HAAR_FACE_CASCADE_URL = (
    "https://raw.githubusercontent.com/opencv/opencv/4.x/data/haarcascades/"
    "haarcascade_frontalface_default.xml"
)
HAAR_EYE_CASCADE_URL = (
    "https://raw.githubusercontent.com/opencv/opencv/4.x/data/haarcascades/"
    "haarcascade_eye.xml"
)
DEFAULT_HAAR_FACE_PATH = Path.home() / ".cache" / "drivesense" / "haarcascade_frontalface_default.xml"
DEFAULT_HAAR_EYE_PATH = Path.home() / ".cache" / "drivesense" / "haarcascade_eye.xml"

# Coarse open/closed EAR-proxy values. Chosen so they land clearly on either
# side of a typical MediaPipe-based closed-eye threshold (~0.2, see
# `drivesense.inference.drowsiness`) without claiming false precision.
HAAR_EAR_OPEN = 0.30
HAAR_EAR_HALF = 0.20
HAAR_EAR_CLOSED = 0.05

BBox = Tuple[int, int, int, int]  # x, y, w, h


def download_haar_cascade(url: str, dest: Path) -> Path:
    """Download a Haar cascade XML file if not already cached locally."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        urllib.request.urlretrieve(url, dest)
    return dest


def features_from_face_and_eyes(
    face_bbox: Optional[BBox],
    eye_bboxes: Sequence[BBox],
    frame_width: int,
) -> FrameLandmarkFeatures:
    """Pure-logic core of the Haar fallback: bounding boxes -> features.

    Kept separate from `HaarCascadeLandmarkExtractor.extract()` so it is
    unit-testable without OpenCV cascade files/network access, the same
    separation `_estimate_head_pose` already gives the MediaPipe path.
    """
    if face_bbox is None:
        return _no_face_result()

    n_eyes = min(len(eye_bboxes), 2)
    if n_eyes == 2:
        ear_proxy = HAAR_EAR_OPEN
    elif n_eyes == 1:
        ear_proxy = HAAR_EAR_HALF
    else:
        ear_proxy = HAAR_EAR_CLOSED

    fx, _fy, fw, _fh = face_bbox
    face_center_x = fx + fw / 2.0
    frame_center_x = frame_width / 2.0
    # Same directional convention as `_estimate_head_pose`: positive yaw ==
    # face/nose shifted toward the right of frame.
    yaw_deg = ((face_center_x - frame_center_x) / max(frame_width / 2.0, 1e-6)) * 45.0

    return FrameLandmarkFeatures(
        left_ear=ear_proxy,
        right_ear=ear_proxy,
        mar=0.0,  # unavailable via Haar cascades in this build — see module note above
        head_pitch_deg=0.0,
        head_yaw_deg=yaw_deg,
        head_roll_deg=0.0,
        face_detected=True,
    )


class HaarCascadeLandmarkExtractor:
    """Lightweight local fallback implementing the `LandmarkExtractor` protocol.

    Real detection (genuinely runs `cv2.CascadeClassifier.detectMultiScale`
    on each frame), but a coarse approximation of true landmark-based
    EAR/MAR — see the module-level note above before using its output
    anywhere results are reported as if MediaPipe-equivalent.
    """

    def __init__(
        self,
        face_cascade_path: Optional[Path] = None,
        eye_cascade_path: Optional[Path] = None,
    ) -> None:
        import cv2

        face_cascade_path = face_cascade_path or download_haar_cascade(
            HAAR_FACE_CASCADE_URL, DEFAULT_HAAR_FACE_PATH
        )
        eye_cascade_path = eye_cascade_path or download_haar_cascade(
            HAAR_EYE_CASCADE_URL, DEFAULT_HAAR_EYE_PATH
        )
        self._cv2 = cv2
        self._face_cascade = cv2.CascadeClassifier(str(face_cascade_path))
        self._eye_cascade = cv2.CascadeClassifier(str(eye_cascade_path))
        if self._face_cascade.empty() or self._eye_cascade.empty():
            raise RuntimeError(
                "Failed to load Haar cascade XML files "
                f"({face_cascade_path}, {eye_cascade_path}) — they may be "
                "corrupt or incompletely downloaded; delete and retry."
            )

    def extract(self, frame_rgb) -> FrameLandmarkFeatures:
        """Extract coarse open/closed-eye + yaw-proxy features from one RGB frame."""
        cv2 = self._cv2
        gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
        frame_height, frame_width = gray.shape[:2]

        faces = self._face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
        if len(faces) == 0:
            return _no_face_result()

        # Largest detected face, in case of spurious secondary detections.
        fx, fy, fw, fh = max(faces, key=lambda b: b[2] * b[3])
        # Eyes sit in the upper ~60% of the face box; restricting the search
        # region reduces false eye-like detections from mouth/nostril texture.
        eye_region = gray[fy : fy + int(fh * 0.6), fx : fx + fw]
        eyes = self._eye_cascade.detectMultiScale(eye_region, scaleFactor=1.1, minNeighbors=8, minSize=(15, 15))

        return features_from_face_and_eyes(
            face_bbox=(int(fx), int(fy), int(fw), int(fh)),
            eye_bboxes=[tuple(int(v) for v in e) for e in eyes],
            frame_width=frame_width,
        )
