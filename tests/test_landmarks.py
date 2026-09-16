"""Tests for the parts of drivesense.features.landmarks that don't require
constructing the full MediaPipe FaceLandmarker graph.

NOTE: constructing `MediaPipeLandmarkExtractor` itself is intentionally NOT
tested here. On this development machine it was observed to be killed by
the OS (exit 137, consistent with OOM) due to desktop memory pressure — see
the class docstring and docs/Master_Plan_Status.md. A crash like that would
take down the whole pytest process, not just fail one test, so it is
verified manually via scripts/verify_landmarks_extractor.py instead (run in
Colab or when enough local memory is free), not in the automated suite.
"""

from types import SimpleNamespace

from drivesense.features.landmarks import (
    LEFT_EYE_IDX,
    OUTER_MOUTH_IDX,
    RIGHT_EYE_IDX,
    FrameLandmarkFeatures,
    _estimate_head_pose,
    _no_face_result,
    download_face_landmarker_model,
)


def _fake_landmarks(n: int = 468):
    # Enough points at (0.5, 0.5) so any index access is valid; override the
    # few indices _estimate_head_pose actually reads.
    points = [SimpleNamespace(x=0.5, y=0.5, z=0.0) for _ in range(n)]
    points[33] = SimpleNamespace(x=0.3, y=0.5, z=0.0)   # left eye outer
    points[263] = SimpleNamespace(x=0.7, y=0.5, z=0.0)  # right eye outer
    points[1] = SimpleNamespace(x=0.5, y=0.5, z=0.0)    # nose tip (centered -> yaw ~ 0)
    return points


def test_no_face_result_shape():
    result = _no_face_result()
    assert isinstance(result, FrameLandmarkFeatures)
    assert result.face_detected is False
    assert result.left_ear == 0.0 and result.right_ear == 0.0 and result.mar == 0.0


def test_estimate_head_pose_centered_nose_gives_near_zero_yaw():
    landmarks = _fake_landmarks()
    pitch, yaw, roll = _estimate_head_pose(landmarks)
    assert pitch == 0.0  # not yet estimated, by design
    assert abs(yaw) < 1e-6
    assert abs(roll) < 1e-6  # eyes level -> ~0 roll


def test_estimate_head_pose_nose_shifted_right_gives_positive_yaw():
    landmarks = _fake_landmarks()
    landmarks[1] = SimpleNamespace(x=0.65, y=0.5, z=0.0)  # nose shifted toward right eye
    _, yaw, _ = _estimate_head_pose(landmarks)
    assert yaw > 0


def test_landmark_index_tuples_have_six_points_each():
    assert len(LEFT_EYE_IDX) == 6
    assert len(RIGHT_EYE_IDX) == 6
    assert len(OUTER_MOUTH_IDX) == 6


def test_download_face_landmarker_model_is_idempotent(tmp_path):
    dest = tmp_path / "face_landmarker.task"
    path1 = download_face_landmarker_model(dest)
    mtime1 = path1.stat().st_mtime
    path2 = download_face_landmarker_model(dest)  # should not re-download
    mtime2 = path2.stat().st_mtime
    assert path1 == path2 == dest
    assert mtime1 == mtime2
    assert dest.stat().st_size > 0
