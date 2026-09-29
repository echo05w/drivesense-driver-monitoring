"""Unit tests for drivesense.inference.risk_engine."""

import pytest

from drivesense.inference.risk_engine import (
    AwarenessRiskLevel,
    DriverAwarenessConfig,
    DriverAwarenessEngine,
    RiskEngineConfig,
    RiskLevel,
    TemporalRiskEngine,
)


def test_starts_at_normal():
    engine = TemporalRiskEngine()
    assert engine._current_level() == RiskLevel.NORMAL


def test_sustained_distraction_escalates_to_warning():
    engine = TemporalRiskEngine(RiskEngineConfig(window_size=10))
    level = RiskLevel.NORMAL
    for _ in range(10):
        level = engine.update(distraction_label="texting", drowsiness_label=None)
    assert level == RiskLevel.WARNING


def test_occasional_distraction_stays_normal():
    engine = TemporalRiskEngine(RiskEngineConfig(window_size=10, caution_fraction=0.3))
    level = RiskLevel.NORMAL
    for i in range(10):
        label = "texting" if i == 0 else None
        level = engine.update(distraction_label=label, drowsiness_label=None)
    assert level == RiskLevel.NORMAL


def test_drowsy_label_triggers_escalation():
    engine = TemporalRiskEngine(RiskEngineConfig(window_size=5, warning_fraction=0.6))
    level = RiskLevel.NORMAL
    for _ in range(5):
        level = engine.update(distraction_label="safe_driving", drowsiness_label="drowsy")
    assert level == RiskLevel.WARNING


def test_risk_decays_after_evidence_clears():
    engine = TemporalRiskEngine(RiskEngineConfig(window_size=5, warning_fraction=0.6, caution_fraction=0.3))
    for _ in range(5):
        engine.update(distraction_label="texting", drowsiness_label=None)
    assert engine._current_level() == RiskLevel.WARNING
    level = RiskLevel.WARNING
    for _ in range(5):
        level = engine.update(distraction_label=None, drowsiness_label=None)
    assert level == RiskLevel.NORMAL


def test_reset_clears_history():
    engine = TemporalRiskEngine(RiskEngineConfig(window_size=5))
    for _ in range(5):
        engine.update(distraction_label="texting", drowsiness_label=None)
    engine.reset()
    assert engine._current_level() == RiskLevel.NORMAL


# --- DriverAwarenessEngine (continuous fusion) ---------------------------


def test_awareness_config_rejects_weights_not_summing_to_one():
    with pytest.raises(ValueError):
        DriverAwarenessConfig(drowsiness_weight=0.5, distraction_weight=0.5, temporal_weight=0.5)


def test_awareness_all_clear_is_low_risk_high_awareness():
    engine = DriverAwarenessEngine()
    result = None
    for _ in range(10):
        result = engine.update(distraction_score=0.0, drowsiness_score=0.0)
    assert result["risk_level"] == AwarenessRiskLevel.LOW
    assert result["awareness"] > 0.9
    assert result["risk_score"] < 0.1


def test_sustained_severe_drowsiness_and_distraction_reaches_critical():
    engine = DriverAwarenessEngine()
    result = None
    for _ in range(30):
        result = engine.update(distraction_score=1.0, drowsiness_score=1.0)
    assert result["risk_level"] == AwarenessRiskLevel.CRITICAL
    assert result["drowsiness"] > 0.9
    assert result["awareness"] < 0.1


def test_missing_drowsiness_signal_reports_none_not_fabricated_zero():
    engine = DriverAwarenessEngine()
    result = engine.update(distraction_score=0.2, drowsiness_score=None)
    assert result["drowsiness"] is None
    # Distraction-only evidence still contributes to risk_score.
    assert result["risk_score"] > 0.0


def test_single_noisy_frame_does_not_flip_to_critical():
    engine = DriverAwarenessEngine()
    for _ in range(20):
        engine.update(distraction_score=0.0, drowsiness_score=0.0)
    result = engine.update(distraction_score=1.0, drowsiness_score=1.0)
    # One noisy frame after a long calm history should not alone reach CRITICAL.
    assert result["risk_level"] != AwarenessRiskLevel.CRITICAL


def test_risk_score_matches_weighted_formula_on_first_frame():
    config = DriverAwarenessConfig(
        drowsiness_weight=0.55, distraction_weight=0.35, temporal_weight=0.10
    )
    engine = DriverAwarenessEngine(config)
    result = engine.update(distraction_score=0.4, drowsiness_score=0.8)
    # First frame: EMA equals the raw input, temporal_penalty from one concerning flag = 1.0.
    expected = 0.55 * 0.8 + 0.35 * 0.4 + 0.10 * 1.0
    assert result["risk_score"] == pytest.approx(expected, abs=1e-6)
