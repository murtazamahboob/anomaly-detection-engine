"""Thresholds from normal validation data — never from the scored data.

The blueprint's rule: the 99th to 99.5th percentile of normal validation
scores. A threshold fitted on scored data would move with the anomalies and
hide exactly the events it should catch.
"""

from __future__ import annotations

import numpy as np


def percentile_threshold(scores: np.ndarray, percentile: float = 99.5) -> float:
    """Score above which a normal-validation point is flagged."""
    scores = np.asarray(scores, dtype=float)
    if len(scores) == 0:
        raise ValueError("scores must not be empty")
    if not 0.0 < percentile < 100.0:
        raise ValueError("percentile must be between 0 and 100")
    return float(np.percentile(scores, percentile))


def mean_plus_std_threshold(scores: np.ndarray, k: float = 3.0) -> float:
    scores = np.asarray(scores, dtype=float)
    if len(scores) == 0:
        raise ValueError("scores must not be empty")
    return float(scores.mean() + k * scores.std())


def flag(scores: np.ndarray, threshold: float) -> np.ndarray:
    """Boolean anomaly flags for scores against a threshold."""
    return np.asarray(scores, dtype=float) > threshold
