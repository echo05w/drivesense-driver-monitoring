"""Tests for UTA-RLDD feature extraction and time-based windowing.

All synthetic - no real video files or MediaPipe graph construction
required, since MediaPipeLandmarkExtractor cannot be constructed on this
development machine (OOM, see docs/LEARNING_LOG.md). `extract_video_features`
accepts any object satisfying the `LandmarkExtractor` protocol, so a fake
extractor exercises the same code path the real Colab run will use.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from drivesense.data.drowsiness import (
    FEATURE_COLUMNS,
    extract_video_features,
    label_from_video_stem,
    make_time_windows,
)
from drivesense.features.landmarks import FrameLandmarkFeatures


# --- label_from_video_stem -------------------------------------------------


def test_label_from_video_stem_known_values():
    assert label_from_video_stem("0") == "alert"
    assert label_from_video_stem("5") == "low_vigilant"
    assert label_from_video_stem("10") == "drowsy"


def test_label_from_video_stem_unknown_raises():
    with pytest.raises(KeyError):
        label_from_video_stem("99")


# --- extract_video_features -------------------------------------------------


class FakeExtractor:
    """Deterministic fake satisfying the LandmarkExtractor protocol: alternates
    face-detected/not, and returns a feature value derived from frame index
    so tests can assert exact values, not just shapes."""

    def extract(self, frame_rgb) -> FrameLandmarkFeatures:
        frame_idx = frame_rgb  # test doubles pass the int frame index directly
        if frame_idx % 5 == 4:  # occasionally no face, to test face_detected tracking
            return FrameLandmarkFeatures(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, face_detected=False)
        return FrameLandmarkFeatures(
            left_ear=0.3 + 0.01 * frame_idx,
            right_ear=0.3 + 0.01 * frame_idx,
            mar=0.1,
            head_pitch_deg=0.0,
            head_yaw_deg=0.0,
            head_roll_deg=0.0,
            face_detected=True,
        )


def _fake_reader_factory(fps: float, n_frames: int):
    def _reader(path):
        def frames():
            for i in range(n_frames):
                yield i, i  # pass frame index as the "frame" for FakeExtractor to use
        return fps, n_frames, frames()
    return _reader


def test_extract_video_features_samples_by_time_not_frame_count():
    # 30fps source, sample_fps=5 -> should keep every 6th frame (30/5=6).
    reader = _fake_reader_factory(fps=30.0, n_frames=60)
    result = extract_video_features(
        video_path="fake.mp4",
        extractor=FakeExtractor(),
        subject="p01",
        label="alert",
        sample_fps=5.0,
        read_frame_rgb_fn=reader,
    )
    kept_frame_indices = [row["frame_idx"] for row in result.frame_rows]
    assert kept_frame_indices == list(range(0, 60, 6))
    assert result.source_fps == 30.0
    assert result.n_frames_processed == len(kept_frame_indices)


def test_extract_video_features_different_fps_yields_different_sampling_but_same_real_rate():
    # 12fps source, sample_fps=5 -> interval = round(12/5) = 2 frames apart,
    # i.e. every ~0.167s, close to the 30fps case's every 0.2s - both are
    # "about 5 samples per real second", not "every Nth frame" blindly.
    reader = _fake_reader_factory(fps=12.0, n_frames=24)
    result = extract_video_features(
        video_path="fake.mp4",
        extractor=FakeExtractor(),
        subject="p01",
        label="alert",
        sample_fps=5.0,
        read_frame_rgb_fn=reader,
    )
    kept_frame_indices = [row["frame_idx"] for row in result.frame_rows]
    assert kept_frame_indices == list(range(0, 24, 2))
    # Real elapsed time between samples should be close to 1/5s regardless of source fps.
    timestamps = [row["timestamp_sec"] for row in result.frame_rows]
    diffs = np.diff(timestamps)
    assert np.allclose(diffs, 2 / 12.0)


def test_extract_video_features_tracks_face_detection_and_timestamps():
    reader = _fake_reader_factory(fps=5.0, n_frames=10)
    result = extract_video_features(
        video_path="fake.mp4",
        extractor=FakeExtractor(),
        subject="p01",
        label="drowsy",
        sample_fps=5.0,  # interval=1, keep every frame
        read_frame_rgb_fn=reader,
    )
    assert result.n_frames_processed == 10
    # Frame 4 and 9 are "no face" per FakeExtractor (idx % 5 == 4).
    assert result.n_faces_detected == 8
    for row in result.frame_rows:
        assert row["subject"] == "p01"
        assert row["label"] == "drowsy"
        assert row["timestamp_sec"] == pytest.approx(row["frame_idx"] / 5.0)
        assert row["face_detected"] == (row["frame_idx"] % 5 != 4)


# --- make_time_windows -------------------------------------------------


def _synthetic_frame_df(subject: str, label: str, duration_sec: float, fps: float, ramp: bool = True) -> pd.DataFrame:
    # linspace (not arange) so the last sample lands exactly at duration_sec,
    # matching how many windows callers expect to fit - arange(int(duration*fps))/fps
    # falls just short of duration_sec due to floor rounding, which is a
    # test-data artifact, not something make_time_windows needs to tolerate.
    n = int(duration_sec * fps) + 1
    t = np.linspace(0, duration_sec, n)
    value = t if ramp else np.full(n, 0.5)
    return pd.DataFrame(
        {
            "subject": subject,
            "label": label,
            "timestamp_sec": t,
            "face_detected": True,
            "left_ear": value,
            "right_ear": value,
            "mar": value,
            "head_pitch_deg": 0.0,
            "head_yaw_deg": 0.0,
            "head_roll_deg": 0.0,
        }
    )


def test_make_time_windows_shape_and_count():
    df = _synthetic_frame_df("p01", "alert", duration_sec=30.0, fps=30.0)
    windows, labels, subjects = make_time_windows(
        df, window_seconds=10.0, stride_seconds=10.0, target_steps=20
    )
    assert windows.shape == (3, 20, len(FEATURE_COLUMNS))  # 30s / 10s stride = 3 windows
    assert (labels == "alert").all()
    assert (subjects == "p01").all()


def test_make_time_windows_resamples_correctly_across_different_fps():
    # Same real signal (a linear ramp from 0 to window_seconds), sampled at
    # two different fps - the resampled window content should match closely
    # regardless of source fps, proving time-based (not frame-based) resampling.
    df_30fps = _synthetic_frame_df("p01", "alert", duration_sec=10.0, fps=30.0)
    df_12fps = _synthetic_frame_df("p02", "alert", duration_sec=10.0, fps=12.0)

    w30, _, _ = make_time_windows(df_30fps, window_seconds=10.0, stride_seconds=10.0, target_steps=10)
    w12, _, _ = make_time_windows(df_12fps, window_seconds=10.0, stride_seconds=10.0, target_steps=10)

    assert w30.shape == w12.shape == (1, 10, len(FEATURE_COLUMNS))
    np.testing.assert_allclose(w30[0], w12[0], atol=0.15)


def test_make_time_windows_never_crosses_subject_or_label_boundary():
    df = pd.concat(
        [
            _synthetic_frame_df("p01", "alert", duration_sec=10.0, fps=30.0, ramp=False),
            _synthetic_frame_df("p01", "drowsy", duration_sec=10.0, fps=30.0, ramp=False),
            _synthetic_frame_df("p02", "alert", duration_sec=10.0, fps=30.0, ramp=False),
        ],
        ignore_index=True,
    )
    windows, labels, subjects = make_time_windows(
        df, window_seconds=5.0, stride_seconds=5.0, target_steps=10
    )
    # 3 independent (subject, label) groups x 2 windows each (10s / 5s stride) = 6
    assert len(windows) == 6
    for lbl, subj in zip(labels, subjects):
        assert (subj, lbl) in {("p01", "alert"), ("p01", "drowsy"), ("p02", "alert")}


def test_make_time_windows_drops_low_face_detection_windows():
    df = _synthetic_frame_df("p01", "alert", duration_sec=10.0, fps=30.0, ramp=False)
    df.loc[df["timestamp_sec"] < 5.0, "face_detected"] = False  # first window mostly no-face
    windows, labels, subjects = make_time_windows(
        df, window_seconds=5.0, stride_seconds=5.0, target_steps=10, require_face_fraction=0.5
    )
    assert len(windows) == 1  # only the second (all-face) window survives


def test_make_time_windows_empty_input_returns_correctly_shaped_empty_arrays():
    df = pd.DataFrame(columns=["subject", "label", "timestamp_sec", "face_detected", *FEATURE_COLUMNS])
    windows, labels, subjects = make_time_windows(df, window_seconds=5.0, stride_seconds=5.0, target_steps=10)
    assert windows.shape == (0, 10, len(FEATURE_COLUMNS))
    assert len(labels) == 0
    assert len(subjects) == 0


def test_make_time_windows_rejects_invalid_params():
    df = _synthetic_frame_df("p01", "alert", duration_sec=10.0, fps=30.0)
    with pytest.raises(ValueError):
        make_time_windows(df, window_seconds=0, stride_seconds=5.0, target_steps=10)
    with pytest.raises(ValueError):
        make_time_windows(df, window_seconds=5.0, stride_seconds=-1, target_steps=10)
    with pytest.raises(ValueError):
        make_time_windows(df, window_seconds=5.0, stride_seconds=5.0, target_steps=1)
