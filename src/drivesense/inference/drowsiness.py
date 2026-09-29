"""Rule-based drowsiness scoring from landmark features (PERCLOS-style).

No trained drowsiness model exists yet (see `docs/Master_Plan_Status.md` row
12 — the temporal `DrowsinessGRU`/`DrowsinessTemporalCNN` architectures in
`drivesense.models.temporal` are implemented and smoke-tested but never
trained on real data, since real MediaPipe feature extraction over UTA-RLDD
has not been executed: blocked locally by OOM, deferred to
`notebooks/02_Drowsiness_Colab_Pipeline.ipynb`). This module is therefore a
genuine, real-time-computable *rule-based* stand-in, not a placeholder
pretending to be the trained model: it implements PERCLOS (PERcentage of eye
CLOSure), an established real drowsiness metric from the fatigue-detection
literature (Wierwille & Ellsworth, 1994), over a rolling time window of
per-frame EAR (and, when available, MAR) values, exactly the kind of
"strongest feasible pipeline using existing resources" the project's rules
call for while the trained model is blocked.

`PerclosDrowsinessScorer` is what actually powers `scripts/demo.py` today.
It accepts any `drivesense.features.landmarks.FrameLandmarkFeatures`,
whether produced by the real `MediaPipeLandmarkExtractor` (Colab) or the
coarser local `HaarCascadeLandmarkExtractor` fallback — the score is honest
about which input it got via `PerclosConfig`, but does not itself know or
care which extractor produced the features.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional

from drivesense.features.landmarks import FrameLandmarkFeatures


@dataclass
class PerclosConfig:
    window_size: int = 60  # frames considered for the rolling closure/yawn fraction
    ear_closed_threshold: float = 0.21  # below this counts as "eyes closed" this frame
    mar_yawn_threshold: float = 0.6  # above this counts as "yawning" this frame
    yawn_weight: float = 0.3  # how much the yawn fraction adds on top of eye closure


class PerclosDrowsinessScorer:
    """Rolling PERCLOS-based drowsiness score in [0, 1] from landmark features."""

    def __init__(self, config: Optional[PerclosConfig] = None) -> None:
        self.config = config or PerclosConfig()
        self._closed_flags: Deque[bool] = deque(maxlen=self.config.window_size)
        self._yawn_flags: Deque[bool] = deque(maxlen=self.config.window_size)

    def reset(self) -> None:
        self._closed_flags.clear()
        self._yawn_flags.clear()

    def update(self, features: FrameLandmarkFeatures) -> Optional[float]:
        """Feed one frame's landmark features, return the current score.

        Returns None while no data point has ever landed in the current
        window (e.g. no face seen yet) rather than fabricating a 0 — a
        genuinely unknown drowsiness state must not be reported as "0% (not
        drowsy)", which would be a false negative claim, not an absence of
        evidence.
        """
        if not features.face_detected:
            return self._score()

        avg_ear = (features.left_ear + features.right_ear) / 2.0
        self._closed_flags.append(avg_ear < self.config.ear_closed_threshold)
        self._yawn_flags.append(features.mar > self.config.mar_yawn_threshold)
        return self._score()

    def _score(self) -> Optional[float]:
        if not self._closed_flags:
            return None
        eye_closure_fraction = sum(self._closed_flags) / len(self._closed_flags)
        yawn_fraction = sum(self._yawn_flags) / len(self._yawn_flags) if self._yawn_flags else 0.0
        score = eye_closure_fraction + self.config.yawn_weight * yawn_fraction
        return max(0.0, min(1.0, score))
