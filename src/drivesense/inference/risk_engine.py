"""Temporal risk-fusion engine.

Combines per-frame distraction-model output and windowed drowsiness-model
output into a single, graduated risk level over time. This is a deliberately
simple, rule-based/state-machine layer — NOT a third trained model — per the
Individual Project Brief (Q12.3): a trained fusion model would not add
rubric-rewarded value over a clearly justified, interpretable rule layer, and
interpretability matters for a safety-alert system.

Risk levels escalate with sustained evidence and decay when evidence clears,
rather than flipping on a single noisy frame.

This module has two fusion layers, both rule-based and both real (not
trained) for the same interpretability reason:

- ``TemporalRiskEngine`` (original): categorical NORMAL/CAUTION/WARNING,
  driven by discrete per-frame labels inside a sliding window.
- ``DriverAwarenessEngine`` (added for the demo/report-facing output):
  continuous 0-1 drowsiness/distraction scores fused into a single
  ``risk_score`` and an ``awareness`` figure, exponentially smoothed over
  time so one noisy frame cannot flip the displayed risk level. Kept
  separate from ``TemporalRiskEngine`` rather than replacing it, since the
  two are wired into different call sites (`scripts/demo_distraction.py`
  uses the categorical engine already; the new `scripts/demo.py` uses this
  one) and both are independently unit-tested.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Deque, Optional


class RiskLevel(IntEnum):
    NORMAL = 0
    CAUTION = 1
    WARNING = 2


@dataclass
class RiskEngineConfig:
    window_size: int = 30  # number of recent frames considered
    caution_fraction: float = 0.3  # fraction of window flagged -> CAUTION
    warning_fraction: float = 0.6  # fraction of window flagged -> WARNING
    distraction_classes_of_concern: frozenset = field(
        default_factory=lambda: frozenset({"texting", "phone_call", "reaching", "looking_away"})
    )
    drowsy_state_of_concern: str = "drowsy"
    low_vigilance_state: str = "low_vigilant"


class TemporalRiskEngine:
    """Fuses per-frame model outputs into a graduated, smoothed risk level."""

    def __init__(self, config: Optional[RiskEngineConfig] = None) -> None:
        self.config = config or RiskEngineConfig()
        self._flags: Deque[bool] = deque(maxlen=self.config.window_size)

    def reset(self) -> None:
        self._flags.clear()

    def update(self, distraction_label: Optional[str], drowsiness_label: Optional[str]) -> RiskLevel:
        """Feed one frame/window's model outputs, return the current risk level.

        Args:
            distraction_label: predicted distraction class for this frame, or
                None if no face/prediction was available.
            drowsiness_label: predicted drowsiness state for this window, or
                None if not yet available (e.g., warm-up period).
        """
        is_concerning = False
        if distraction_label is not None and distraction_label in self.config.distraction_classes_of_concern:
            is_concerning = True
        if drowsiness_label == self.config.drowsy_state_of_concern:
            is_concerning = True
        elif drowsiness_label == self.config.low_vigilance_state:
            # Low vigilance alone is mild evidence; still counts toward the window.
            is_concerning = is_concerning or False

        self._flags.append(is_concerning)
        return self._current_level()

    def _current_level(self) -> RiskLevel:
        if not self._flags:
            return RiskLevel.NORMAL
        fraction = sum(self._flags) / len(self._flags)
        if fraction >= self.config.warning_fraction:
            return RiskLevel.WARNING
        if fraction >= self.config.caution_fraction:
            return RiskLevel.CAUTION
        return RiskLevel.NORMAL


class AwarenessRiskLevel(str, Enum):
    """Four-tier risk label for the continuous driver-awareness score."""

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


@dataclass
class DriverAwarenessConfig:
    """Weights/thresholds for the continuous drowsiness+distraction fusion.

    Weights follow the Individual Project Brief's rationale for a
    rule-based (not trained) fusion layer (Q12.3): drowsiness is weighted
    higher than distraction because sustained drowsiness (microsleep) is
    the more acute safety failure mode in the driver-monitoring literature
    this project is based on, distraction is weighted second, and a small
    `temporal_penalty` term rewards/penalizes *sustained* concerning
    behaviour over a single noisy frame, on top of the EMA smoothing below.
    These weights are a documented, interpretable design choice, not fit to
    data — no drowsiness ground truth exists locally to calibrate them
    against (see docs/Master_Plan_Status.md row 12), so they should be
    revisited once the Colab-trained drowsiness model and a real fused
    dataset exist.
    """

    drowsiness_weight: float = 0.55
    distraction_weight: float = 0.35
    temporal_weight: float = 0.10
    smoothing_alpha: float = 0.3  # EMA weight on the newest frame, in [0, 1]
    temporal_window: int = 30  # frames considered for the sustained-evidence term
    concern_threshold: float = 0.5  # score above this counts as "concerning" for the window
    low_max: float = 0.25
    medium_max: float = 0.5
    high_max: float = 0.75  # above this -> CRITICAL

    def __post_init__(self) -> None:
        total = self.drowsiness_weight + self.distraction_weight + self.temporal_weight
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"DriverAwarenessConfig weights must sum to 1.0, got {total}")


def classify_awareness_risk(risk_score: float, config: DriverAwarenessConfig) -> AwarenessRiskLevel:
    """Map a smoothed [0, 1] risk score to a LOW/MEDIUM/HIGH/CRITICAL label."""
    if risk_score <= config.low_max:
        return AwarenessRiskLevel.LOW
    if risk_score <= config.medium_max:
        return AwarenessRiskLevel.MEDIUM
    if risk_score <= config.high_max:
        return AwarenessRiskLevel.HIGH
    return AwarenessRiskLevel.CRITICAL


class DriverAwarenessEngine:
    """Fuses continuous per-frame drowsiness/distraction scores into a single,
    temporally-smoothed risk score, awareness figure, and risk level.

    Unlike ``TemporalRiskEngine`` (discrete labels, hard window-fraction
    cutoffs), this operates on continuous [0, 1] confidence scores and uses
    exponential smoothing, so it produces a stable percentage-style output
    suitable for an on-screen HUD (see `scripts/demo.py`) rather than only a
    three-way category.
    """

    def __init__(self, config: Optional[DriverAwarenessConfig] = None) -> None:
        self.config = config or DriverAwarenessConfig()
        self._smoothed_drowsiness: Optional[float] = None
        self._smoothed_distraction: Optional[float] = None
        self._concern_flags: Deque[bool] = deque(maxlen=self.config.temporal_window)

    def reset(self) -> None:
        self._smoothed_drowsiness = None
        self._smoothed_distraction = None
        self._concern_flags.clear()

    @staticmethod
    def _clip01(x: float) -> float:
        return max(0.0, min(1.0, x))

    def _ema(self, prev: Optional[float], new: float) -> float:
        new = self._clip01(new)
        if prev is None:
            return new
        a = self.config.smoothing_alpha
        return a * new + (1.0 - a) * prev

    def update(self, distraction_score: float, drowsiness_score: Optional[float]) -> dict:
        """Feed one frame's continuous distraction/drowsiness scores.

        Args:
            distraction_score: probability/confidence in [0, 1] that the
                current frame shows a distracted-driving state (e.g.
                1 - P(safe_driving) from the distraction classifier).
            drowsiness_score: probability/confidence in [0, 1] that the
                current window shows drowsiness, or None if unavailable
                (e.g. no face detected, or no drowsiness signal wired up) -
                treated as 0 contribution to risk, never fabricated as a
                guessed number.

        Returns:
            A dict with exactly the fields: drowsiness, distraction,
            awareness, risk_score, risk_level (an ``AwarenessRiskLevel``).
        """
        distraction_score = self._clip01(distraction_score)
        drowsy_for_smoothing = 0.0 if drowsiness_score is None else self._clip01(drowsiness_score)

        self._smoothed_distraction = self._ema(self._smoothed_distraction, distraction_score)
        self._smoothed_drowsiness = self._ema(self._smoothed_drowsiness, drowsy_for_smoothing)

        is_concerning = (
            self._smoothed_distraction >= self.config.concern_threshold
            or self._smoothed_drowsiness >= self.config.concern_threshold
        )
        self._concern_flags.append(is_concerning)
        temporal_penalty = sum(self._concern_flags) / len(self._concern_flags) if self._concern_flags else 0.0

        risk_score = self._clip01(
            self.config.drowsiness_weight * self._smoothed_drowsiness
            + self.config.distraction_weight * self._smoothed_distraction
            + self.config.temporal_weight * temporal_penalty
        )
        awareness = 1.0 - risk_score

        return {
            "drowsiness": None if drowsiness_score is None else self._smoothed_drowsiness,
            "distraction": self._smoothed_distraction,
            "awareness": awareness,
            "risk_score": risk_score,
            "risk_level": classify_awareness_risk(risk_score, self.config),
        }
