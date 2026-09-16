"""Engineered-feature baseline classifier.

Per the Individual Project Brief (§7), the baseline for both tasks is a
simple classifier over hand-engineered features (EAR/MAR/head-pose for
drowsiness; could also serve as a sanity-check baseline for distraction if
landmark features are informative there). This intentionally does not use a
deep network — it exists to give the deep models something honest to beat.
"""

from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_logistic_baseline(random_state: int = 42) -> Pipeline:
    """Scaled logistic regression — the simplest honest baseline."""
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=random_state)),
        ]
    )


def build_random_forest_baseline(random_state: int = 42) -> Pipeline:
    """Random forest over engineered features — a slightly stronger baseline
    that does not require feature scaling."""
    return Pipeline(
        steps=[
            ("clf", RandomForestClassifier(n_estimators=200, random_state=random_state)),
        ]
    )
