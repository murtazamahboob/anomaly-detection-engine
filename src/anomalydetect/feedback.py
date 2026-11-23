"""Reviewer feedback: let human corrections move the threshold.

A reviewer marks alerts as real or false. If false alarms exceed the target
false-alarm rate, the threshold rises to the score of the highest-scoring
false alarm (so that exact mistake stops firing); if real anomalies are
being missed, it falls to the lowest-scoring confirmed anomaly. With no
reviewed labels the threshold is unchanged — feedback never invents data.
"""

from __future__ import annotations

import numpy as np


def retune_threshold(scores: np.ndarray, reviewed_labels: np.ndarray,
                     current_threshold: float,
                     target_false_alarm_rate: float = 0.01) -> float:
    """Adjust ``current_threshold`` from reviewed (score, is_real) pairs.

    ``reviewed_labels`` is 1 for a confirmed real anomaly, 0 for a false
    alarm, -1 for unreviewed (ignored).
    """
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(reviewed_labels)
    if len(scores) != len(labels) or len(scores) == 0:
        return float(current_threshold)
    reviewed = labels >= 0
    if not reviewed.any():
        return float(current_threshold)
    false_alarms = reviewed & (labels == 0) & (scores > current_threshold)
    far = float(false_alarms.sum() / max(reviewed.sum(), 1))
    threshold = float(current_threshold)
    if far > target_false_alarm_rate and false_alarms.any():
        # Silence the worst false alarm seen so far.
        threshold = float(max(threshold, scores[false_alarms].max()))
    missed = reviewed & (labels == 1) & (scores <= current_threshold)
    if missed.any():
        threshold = float(min(threshold, scores[missed].min()))
    return threshold


def false_alarm_rate(scores: np.ndarray, labels: np.ndarray,
                     threshold: float) -> float:
    """Share of flagged points that are actually normal."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels)
    flagged = scores > threshold
    if not flagged.any():
        return 0.0
    return float((labels[flagged] == 0).mean())
