"""Unit tests for drivesense.inference.risk_engine."""

from drivesense.inference.risk_engine import RiskEngineConfig, RiskLevel, TemporalRiskEngine


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
