"""One interface for every detector: ``fit / score / explain / predict``.

``AnomalyEngine`` wraps a single detector (``"forest"``, ``"autoencoder"``,
``"lstm"`` or ``"ensemble"``) and owns the threshold: it is fitted on a
held-out slice of normal validation data at the 99.5th percentile, and
``predict`` applies the persistence rule before returning flags, so a lone
noisy window never becomes an alert on its own.
"""

from __future__ import annotations

import numpy as np

from .alerts import apply_persistence, group_alerts
from .autoencoder import Autoencoder
from .ensemble import EnsembleDetector
from .explain import (ae_feature_errors, combined_explanation,
                      forest_feature_contributions, normalise_rows,
                      top_features)
from .features import window_features, window_series
from .forest import IsolationForest
from .lstm_ae import LSTMAutoencoder
from .scaling import StandardScaler
from .threshold import flag, percentile_threshold


class AnomalyEngine:
    def __init__(self, detector: str = "ensemble", percentile: float = 99.5,
                 persistence: int = 2, seed: int = 42):
        if detector not in ("forest", "autoencoder", "lstm", "ensemble"):
            raise ValueError(f"unknown detector: {detector}")
        self.detector = detector
        self.percentile = percentile
        self.persistence = persistence
        self.seed = seed
        self.threshold_: float | None = None
        self.scaler = StandardScaler()
        self._detector = None
        self.feature_names_: list[str] | None = None

    # -- fitting ----------------------------------------------------------
    def fit(self, X: np.ndarray, X_val: np.ndarray | None = None,
            feature_names: list[str] | None = None) -> "AnomalyEngine":
        X = np.asarray(X, dtype=float)
        if X_val is None:
            split = int(0.85 * len(X))
            X_tr, X_val = X[:split], X[split:]
        else:
            X_tr = X
        self.feature_names_ = feature_names
        if self.detector == "ensemble":
            self._detector = EnsembleDetector(seed=self.seed)
            self._detector.fit(X_tr, X_val=X_val)
            val_scores = self._detector.score(X_val)
        elif self.detector == "lstm":
            windows = window_series(X_tr)
            val_windows = window_series(X_val)
            self.scaler.fit(X_tr)
            scaled_windows = self.scaler.transform(
                windows.reshape(-1, windows.shape[2])).reshape(windows.shape)
            scaled_val_windows = self.scaler.transform(
                val_windows.reshape(-1, val_windows.shape[2])).reshape(val_windows.shape)
            self._detector = LSTMAutoencoder(seed=self.seed)
            self._detector.fit(scaled_windows, X_val=scaled_val_windows)
            val_scores = self._detector.reconstruction_error(scaled_val_windows)
        else:
            scaled_tr = self.scaler.fit_transform(X_tr)
            scaled_val = self.scaler.transform(X_val)
            if self.detector == "forest":
                self._detector = IsolationForest(seed=self.seed)
                self._detector.fit(scaled_tr)
                val_scores = self._detector.anomaly_score(scaled_val)
            else:
                self._detector = Autoencoder(seed=self.seed)
                self._detector.fit(scaled_tr, X_val=scaled_val)
                val_scores = self._detector.reconstruction_error(scaled_val)
        self.threshold_ = percentile_threshold(val_scores, self.percentile)
        return self

    # -- scoring ----------------------------------------------------------
    def score(self, X: np.ndarray) -> np.ndarray:
        if self._detector is None or self.threshold_ is None:
            raise ValueError("engine is not fitted")
        X = np.asarray(X, dtype=float)
        if self.detector == "ensemble":
            return self._detector.score(X)
        if self.detector == "lstm":
            windows = window_series(X)
            scaled = self.scaler.transform(
                windows.reshape(-1, windows.shape[2])).reshape(windows.shape)
            return self._detector.reconstruction_error(scaled)
        scaled = self.scaler.transform(X)
        if self.detector == "forest":
            return self._detector.anomaly_score(scaled)
        return self._detector.reconstruction_error(scaled)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Flags after the persistence rule; ``score`` is the raw signal."""
        return apply_persistence(flag(self.score(X), self.threshold_),
                                 self.persistence)

    def incidents(self, X: np.ndarray, times_minutes: np.ndarray) -> list[dict]:
        return group_alerts(times_minutes, self.predict(X))

    # -- explanation ------------------------------------------------------
    def explain(self, X: np.ndarray) -> np.ndarray:
        """Per-sample, per-feature blame shares (rows sum to 1)."""
        if self._detector is None:
            raise ValueError("engine is not fitted")
        X = np.asarray(X, dtype=float)
        if self.detector == "ensemble":
            # The ensemble owns its own scaler (fit inside the detector).
            scaled = self._detector.scaler.transform(X)
        else:
            scaled = self.scaler.transform(X)
        if self.detector == "forest":
            return forest_feature_contributions(self._detector, scaled)
        if self.detector == "autoencoder":
            return normalise_rows(
                self._detector.feature_error(scaled))
        if self.detector == "ensemble":
            ae_err = self._detector.ae.feature_error(scaled)
            forest_c = forest_feature_contributions(
                self._detector.forest, scaled)
            return combined_explanation(ae_err, forest_c)
        # LSTM: explain windows by per-feature reconstruction error.
        windows = window_series(X)
        scaled_w = self.scaler.transform(
            windows.reshape(-1, windows.shape[2])).reshape(windows.shape)
        recon = self._detector.reconstruct(scaled_w)
        per_feature = ((recon - scaled_w) ** 2).mean(axis=1)
        return normalise_rows(per_feature)

    def top_features(self, X: np.ndarray, k: int = 3) -> list[list[tuple[str, float]]]:
        names = self.feature_names_ or [f"feature_{i}" for i in range(np.asarray(X).shape[1])]
        return top_features(self.explain(X), names, k=k)
