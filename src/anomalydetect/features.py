"""Windowing and feature extraction for time series, logs and prices.

Detectors score rows. For a time series, a row is either one timestep or
one window summarised into features — the blueprint's pipeline step 3.
All feature builders are pure functions of the input, so a scored window
never depends on statistics from other scored data.
"""

from __future__ import annotations

import numpy as np


def window_series(X: np.ndarray, window: int = 20, stride: int = 10) -> np.ndarray:
    """Cut ``(n, d)`` into overlapping ``(n_windows, window, d)`` windows."""
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must be a 2-D array")
    if window < 2 or stride < 1:
        raise ValueError("window must be >= 2 and stride >= 1")
    starts = range(0, len(X) - window + 1, stride)
    return np.stack([X[s:s + window] for s in starts])


def window_features(windows: np.ndarray) -> np.ndarray:
    """Per-window aggregates: mean, std, min, max, range, slope per channel.

    Returns ``(n_windows, 6 * d)`` — a tabular view of each window that the
    forest and the MLP autoencoder can score directly.
    """
    windows = np.asarray(windows, dtype=float)
    if windows.ndim != 3:
        raise ValueError("windows must be (n_windows, T, d)")
    mean = windows.mean(axis=1)
    std = windows.std(axis=1)
    lo = windows.min(axis=1)
    hi = windows.max(axis=1)
    rng = hi - lo
    slope = (windows[:, -1, :] - windows[:, 0, :]) / max(windows.shape[1] - 1, 1)
    return np.concatenate([mean, std, lo, hi, rng, slope], axis=1)


def window_labels(labels: np.ndarray, window: int = 20, stride: int = 10) -> np.ndarray:
    """A window is anomalous if any of its rows is anomalous."""
    labels = np.asarray(labels)
    starts = range(0, len(labels) - window + 1, stride)
    return np.array([int(labels[s:s + window].any()) for s in starts])
