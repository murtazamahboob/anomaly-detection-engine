"""Ensemble: Isolation Forest + MLP autoencoder, rank-averaged.

The two detectors see different things — the forest catches isolated
outliers in feature space, the autoencoder catches broken structure — so
their scores are combined by rank, not by raw value: each score is replaced
by its percentile rank on normal validation data, then averaged. A raw
average would let whichever score has the bigger scale win; ranks cannot.
The ensemble also reports a majority vote: how many of its members flag a
point.
"""

from __future__ import annotations

import numpy as np

from .autoencoder import Autoencoder
from .forest import IsolationForest
from .scaling import StandardScaler


def _rank01(scores: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Percentile rank of each score within the reference scores, in [0, 1]."""
    ref = np.sort(np.asarray(reference, dtype=float))
    scores = np.asarray(scores, dtype=float)
    ranks = np.searchsorted(ref, scores, side="right") / max(len(ref), 1)
    return np.clip(ranks, 0.0, 1.0)


class EnsembleDetector:
    """Forest + autoencoder over the same scaled features."""

    def __init__(self, n_estimators: int = 300, max_samples: int = 256,
                 bottleneck: int = 8, seed: int = 42):
        self.forest = IsolationForest(n_estimators=n_estimators,
                                      max_samples=max_samples, seed=seed)
        self.ae = Autoencoder(bottleneck=bottleneck, seed=seed)
        self.scaler = StandardScaler()
        self.val_forest_: np.ndarray | None = None
        self.val_ae_: np.ndarray | None = None

    def fit(self, X: np.ndarray, X_val: np.ndarray | None = None,
            ae_epochs: int = 120) -> "EnsembleDetector":
        X = np.asarray(X, dtype=float)
        if X_val is None:
            split = int(0.85 * len(X))
            X_tr, X_val = X[:split], X[split:]
        else:
            X_tr = X
        scaled_tr = self.scaler.fit_transform(X_tr)
        scaled_val = self.scaler.transform(X_val)
        self.forest.fit(scaled_tr)
        self.ae.fit(scaled_tr, X_val=scaled_val, epochs=ae_epochs)
        self.val_forest_ = self.forest.anomaly_score(scaled_val)
        self.val_ae_ = self.ae.reconstruction_error(scaled_val)
        return self

    def _scores(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        scaled = self.scaler.transform(X)
        return (self.forest.anomaly_score(scaled),
                self.ae.reconstruction_error(scaled))

    def score(self, X: np.ndarray) -> np.ndarray:
        """Rank-averaged ensemble score in [0, 1]; higher = more anomalous."""
        if self.val_forest_ is None or self.val_ae_ is None:
            raise ValueError("ensemble is not fitted")
        forest_s, ae_s = self._scores(X)
        return 0.5 * (_rank01(forest_s, self.val_forest_)
                      + _rank01(ae_s, self.val_ae_))

    def member_scores(self, X: np.ndarray) -> dict[str, np.ndarray]:
        forest_s, ae_s = self._scores(X)
        return {"isolation_forest": forest_s, "autoencoder": ae_s}

    def votes(self, X: np.ndarray, forest_threshold: float,
              ae_threshold: float) -> np.ndarray:
        """How many of the two members flag each point (0, 1 or 2)."""
        forest_s, ae_s = self._scores(X)
        return ((forest_s > forest_threshold).astype(int)
                + (ae_s > ae_threshold).astype(int))
