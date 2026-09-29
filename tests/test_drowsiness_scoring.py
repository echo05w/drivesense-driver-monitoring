"""Unit tests for drivesense.inference.drowsiness (rule-based PERCLOS scorer)."""

from drivesense.features.landmarks import FrameLandmarkFeatures
from drivesense.inference.drowsiness import PerclosConfig, PerclosDrowsinessScorer


def _features(ear: float, mar: float = 0.0, face_detected: bool = True) -> FrameLandmarkFeatures:
    return FrameLandmarkFeatures(
        left_ear=ear,
        right_ear=ear,
        mar=mar,
        head_pitch_deg=0.0,
        head_yaw_deg=0.0,
        head_roll_deg=0.0,
        face_detected=face_detected,
    )


def test_no_face_ever_seen_returns_none():
    scorer = PerclosDrowsinessScorer()
    result = scorer.update(_features(ear=0.3, face_detected=False))
    assert result is None


def test_eyes_open_gives_low_score():
    scorer = PerclosDrowsinessScorer()
    result = None
    for _ in range(30):
        result = scorer.update(_features(ear=0.35))
    assert result == 0.0


def test_sustained_closed_eyes_gives_high_score():
    config = PerclosConfig(window_size=10, ear_closed_threshold=0.21)
    scorer = PerclosDrowsinessScorer(config)
    result = None
    for _ in range(10):
        result = scorer.update(_features(ear=0.05))
    assert result == 1.0


def test_yawning_adds_to_score_on_top_of_open_eyes():
    config = PerclosConfig(window_size=10, mar_yawn_threshold=0.6, yawn_weight=0.3)
    scorer = PerclosDrowsinessScorer(config)
    result = None
    for _ in range(10):
        result = scorer.update(_features(ear=0.35, mar=0.8))
    assert result == 0.3


def test_reset_clears_history():
    scorer = PerclosDrowsinessScorer(PerclosConfig(window_size=5))
    for _ in range(5):
        scorer.update(_features(ear=0.05))
    scorer.reset()
    assert scorer.update(_features(ear=0.35, face_detected=False)) is None
