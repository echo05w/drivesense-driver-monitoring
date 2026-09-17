"""UTA-RLDD drowsiness feature extraction and time-based windowing.

The MediaPipe-dependent per-frame extraction (`extract_video_features`)
cannot be exercised end-to-end on this development machine — constructing
`MediaPipeLandmarkExtractor` gets OOM-killed here (see
`docs/Master_Plan_Status.md` and `docs/LEARNING_LOG.md`) — but it accepts
any object satisfying `drivesense.features.landmarks.LandmarkExtractor`
(a single `extract(frame_rgb) -> FrameLandmarkFeatures` method), so its
logic is fully unit-testable with a fake extractor and synthetic frames,
independent of that blocker. The real MediaPipe extractor is only
constructed in Google Colab, where this same function is called unmodified.

`make_time_windows` addresses a real, verified finding from this project's
own EDA: UTA-RLDD videos vary widely in frame rate (12-30fps observed
across just 6 subjects) and resolution. Windowing by raw frame count would
silently mix different real-world durations across subjects; this instead
windows by wall-clock time and resamples each window to a fixed number of
time steps via linear interpolation, so every window represents the same
real duration regardless of the source video's frame rate.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from drivesense.features.landmarks import FrameLandmarkFeatures, LandmarkExtractor

FEATURE_COLUMNS = [
    "left_ear",
    "right_ear",
    "mar",
    "head_pitch_deg",
    "head_yaw_deg",
    "head_roll_deg",
]

# UTA-RLDD's own labeling convention (self-reported, Karolinska-Sleepiness-
# -Scale-inspired): each subject records three videos named by their
# self-reported state. This mapping is documented dataset convention, not
# guessed - see docs/Dataset_Research.md and the dataset's own site.
VIDEO_LABEL_MAP: Dict[str, str] = {
    "0": "alert",
    "5": "low_vigilant",
    "10": "drowsy",
}
LABEL_ORDER = ["alert", "low_vigilant", "drowsy"]


def label_from_video_stem(stem: str) -> str:
    """Map a UTA-RLDD video filename stem ("0", "5", "10") to its label.

    Raises KeyError loudly on an unrecognized stem rather than guessing -
    a wrong assumption about file naming should fail immediately, not
    silently mislabel data.
    """
    if stem not in VIDEO_LABEL_MAP:
        raise KeyError(
            f"Unrecognized UTA-RLDD video filename stem '{stem}' - expected one of "
            f"{list(VIDEO_LABEL_MAP)}. Check the actual downloaded filename."
        )
    return VIDEO_LABEL_MAP[stem]


@dataclass
class VideoExtractionResult:
    frame_rows: List[dict]
    n_frames_processed: int
    n_faces_detected: int
    source_fps: float
    source_frame_count: int


def extract_video_features(
    video_path: Path,
    extractor: LandmarkExtractor,
    subject: str,
    label: str,
    sample_fps: float = 5.0,
    read_frame_rgb_fn=None,
) -> VideoExtractionResult:
    """Extract per-sampled-frame landmark features from one video.

    Samples frames at approximately `sample_fps` **real seconds** apart,
    regardless of the source video's native frame rate - this is what makes
    downstream windowing genuinely time-based rather than an artifact of
    whatever fps a given subject's phone happened to record at.

    `read_frame_rgb_fn` is injectable for testing (a fake video reader that
    doesn't need OpenCV/a real file); when None, uses `cv2.VideoCapture`.
    """
    if read_frame_rgb_fn is None:
        import cv2

        def _cv2_reader(path: Path):
            cap = cv2.VideoCapture(str(path))
            fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if fps <= 0:
                cap.release()
                raise ValueError(f"Could not read a valid fps for {path}")

            def frames():
                idx = 0
                while True:
                    ret, frame_bgr = cap.read()
                    if not ret:
                        break
                    yield idx, cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                    idx += 1
                cap.release()

            return fps, frame_count, frames()

        read_frame_rgb_fn = _cv2_reader

    source_fps, source_frame_count, frame_iter = read_frame_rgb_fn(video_path)
    sample_interval_frames = max(int(round(source_fps / sample_fps)), 1)

    rows: List[dict] = []
    n_detected = 0
    n_processed = 0
    for frame_idx, frame_rgb in frame_iter:
        if frame_idx % sample_interval_frames != 0:
            continue
        result: FrameLandmarkFeatures = extractor.extract(frame_rgb)
        timestamp_sec = frame_idx / source_fps
        rows.append(
            {
                "subject": subject,
                "label": label,
                "frame_idx": frame_idx,
                "timestamp_sec": timestamp_sec,
                "face_detected": result.face_detected,
                "left_ear": result.left_ear,
                "right_ear": result.right_ear,
                "mar": result.mar,
                "head_pitch_deg": result.head_pitch_deg,
                "head_yaw_deg": result.head_yaw_deg,
                "head_roll_deg": result.head_roll_deg,
            }
        )
        n_processed += 1
        n_detected += int(result.face_detected)

    return VideoExtractionResult(
        frame_rows=rows,
        n_frames_processed=n_processed,
        n_faces_detected=n_detected,
        source_fps=source_fps,
        source_frame_count=source_frame_count,
    )


def make_time_windows(
    df: pd.DataFrame,
    window_seconds: float,
    stride_seconds: float,
    target_steps: int,
    feature_cols: Sequence[str] = tuple(FEATURE_COLUMNS),
    group_cols: Sequence[str] = ("subject", "label"),
    time_col: str = "timestamp_sec",
    require_face_fraction: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Slice a per-frame feature dataframe into fixed-duration, fixed-shape
    windows, resampled by real time rather than raw row count.

    A window never crosses a (subject, label) boundary - i.e. never crosses
    a video, and by construction never crosses a subject. Windows whose
    fraction of face-detected frames falls below `require_face_fraction`
    are dropped (not enough real signal to trust).

    Returns:
        windows: (n_windows, target_steps, n_features) float32 array,
            each window's raw irregular timestamps linearly resampled to
            `target_steps` evenly spaced points spanning `window_seconds`.
        labels: (n_windows,) array of the group's label.
        subjects: (n_windows,) array of the group's subject id.
    """
    if window_seconds <= 0 or stride_seconds <= 0 or target_steps < 2:
        raise ValueError("window_seconds and stride_seconds must be positive; target_steps >= 2")

    windows: List[np.ndarray] = []
    labels: List[str] = []
    subjects: List[str] = []

    for (subject, label), group in df.groupby(list(group_cols), sort=False):
        group = group.sort_values(time_col)
        t = group[time_col].to_numpy()
        if len(t) == 0:
            continue
        t_start, t_end = t[0], t[-1]

        window_start = t_start
        while window_start + window_seconds <= t_end:
            window_end = window_start + window_seconds
            mask = (t >= window_start) & (t <= window_end)
            if mask.sum() < 2:
                window_start += stride_seconds
                continue

            sub = group.loc[mask]
            face_fraction = sub["face_detected"].mean() if "face_detected" in sub else 1.0
            if face_fraction < require_face_fraction:
                window_start += stride_seconds
                continue

            sub_t = sub[time_col].to_numpy()
            query_t = np.linspace(window_start, window_end, target_steps)
            resampled = np.stack(
                [np.interp(query_t, sub_t, sub[col].to_numpy()) for col in feature_cols],
                axis=1,
            ).astype(np.float32)

            windows.append(resampled)
            labels.append(label)
            subjects.append(subject)
            window_start += stride_seconds

    if not windows:
        return (
            np.empty((0, target_steps, len(feature_cols)), dtype=np.float32),
            np.empty((0,), dtype=object),
            np.empty((0,), dtype=object),
        )

    return np.stack(windows), np.array(labels), np.array(subjects)
