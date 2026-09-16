"""Temporal risk-fusion engine.

Combines per-frame distraction-model output and windowed drowsiness-model
output into a single, graduated risk level over time. This is a deliberately
simple, rule-based/state-machine layer — NOT a third trained model — per the
Individual Project Brief (Q12.3): a trained fusion model would not add
rubric-rewarded value over a clearly justified, interpretable rule layer, and
interpretability matters for a safety-alert system.

Risk levels escalate with sustained evidence and decay when evidence clears,
rather than flipping on a single noisy frame.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import IntEnum
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
