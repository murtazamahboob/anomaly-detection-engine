"""Scaling fitted on training data only — never on the data being scored.

Fitting a scaler on scored data leaks the anomalies into the scale and hides
them. Every detector in this package takes a scaler fitted on normal training
data and reuses it for validation and scoring.
"""

from __future__ import annotations

import numpy as np


class StandardScaler:
    """Mean / standard-deviation scaling with a floor on the scale."""

    def __init__(self) -> None:
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "StandardScaler":
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[0] == 0:
            raise ValueError("X must be a non-empty 2-D array")
        self.mean_ = X.mean(axis=0)
        std = X.std(axis=0)
        # A constant feature must not divide by zero or explode on tiny noise.
        self.scale_ = np.where(std < 1e-8, 1.0, std)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise ValueError("scaler is not fitted")
        return (np.asarray(X, dtype=float) - self.mean_) / self.scale_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise ValueError("scaler is not fitted")
        return np.asarray(X, dtype=float) * self.scale_ + self.mean_
