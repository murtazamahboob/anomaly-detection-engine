"""Deterministic synthetic data: normal sensor/log streams + injected anomalies.

Everything is generated from a seed, so the demo and benchmark regenerate
bit-for-bit. Normal data is a 6-channel sensor stream: correlated Gaussian
noise on top of slow sinusoids (a daily cycle and a slow drift), which is
exactly the kind of "normal" a real plant, server room or market feed has.
Anomalies are the four kinds the blueprint's evaluation section names:
spikes, level shifts, drift and dropouts.
"""

from __future__ import annotations

import numpy as np

N_FEATURES = 6
FEATURE_NAMES = ["temp_c", "vibration", "pressure", "flow", "current_a", "humidity"]


def generate_normal(n: int = 1200, seed: int = 42) -> np.ndarray:
    """Return ``(n, 6)`` normal sensor readings."""
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=float)
    base = np.column_stack(
        [
            62.0 + 4.0 * np.sin(t / 90.0) + 0.02 * t / 100.0,
            0.30 + 0.08 * np.sin(t / 55.0 + 1.0),
            101.0 + 2.0 * np.sin(t / 120.0 + 2.0),
            48.0 + 3.0 * np.sin(t / 70.0 + 0.5),
            5.20 + 0.40 * np.sin(t / 80.0 + 3.0),
            41.0 + 5.0 * np.sin(t / 100.0 + 1.5),
        ]
    )
    # Correlated noise: vibration tracks temperature, current tracks flow.
    noise = rng.normal(0.0, 1.0, size=(n, N_FEATURES))
    noise[:, 1] += 0.35 * noise[:, 0]
    noise[:, 4] += 0.30 * noise[:, 3]
    scales = np.array([1.2, 0.05, 0.8, 1.5, 0.25, 2.0])
    return base + noise * scales


def inject_anomalies(
    X: np.ndarray, seed: int = 7
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Inject 4 labelled anomaly segments into a copy of ``X``.

    Returns ``(X_anom, labels, segments)`` where ``labels`` is 0/1 per row and
    each segment is ``{"kind", "start", "end", "feature"}``. Segment positions
    are fixed fractions of the series so the benchmark is stable across runs;
    only the magnitudes are drawn from the seed.
    """
    rng = np.random.default_rng(seed)
    X = X.copy()
    n = len(X)
    labels = np.zeros(n, dtype=int)
    segments: list[dict] = []
    specs = [
        ("spike", int(0.55 * n), int(0.55 * n) + 3, 1),
        ("level_shift", int(0.65 * n), int(0.65 * n) + 25, 0),
        ("drift", int(0.75 * n), int(0.75 * n) + 40, 3),
        ("dropout", int(0.86 * n), int(0.86 * n) + 12, 4),
    ]
    starts = [s[1] for s in specs] + [n]
    for j, (kind, start, end, feat) in enumerate(specs):
        # Segments must not overlap: clamp each end to the next start, so
        # the labelled counts always add up (matters for short series).
        end = min(end, starts[j + 1], n)
        if kind == "spike":
            X[start:end, feat] += rng.normal(8.0, 1.0) * np.array([1.0])
            # A spike hits every channel a little (a real sensor fault does).
            X[start:end, :] += rng.normal(0.0, 0.6, size=(end - start, N_FEATURES))
        elif kind == "level_shift":
            X[start:end, feat] += 6.0
            X[start:end, 1] += 0.35
        elif kind == "drift":
            ramp = np.linspace(0.0, 7.0, end - start)
            X[start:end, feat] += ramp
        elif kind == "dropout":
            X[start:end, feat] = 0.0
            X[start:end, 5] = 0.0
        labels[start:end] = 1
        segments.append(
            {"kind": kind, "start": int(start), "end": int(end), "feature": int(feat)}
        )
    return X, labels, segments


def make_dataset(
    n_train: int = 800, n_test: int = 600, seed: int = 42
) -> dict:
    """Train on pure normal data; test on normal data with injected anomalies."""
    train = generate_normal(n_train, seed=seed)
    test_clean = generate_normal(n_test, seed=seed + 1)
    test, labels, segments = inject_anomalies(test_clean, seed=seed + 2)
    return {
        "train": train,
        "test": test,
        "test_labels": labels,
        "segments": segments,
        "feature_names": FEATURE_NAMES,
    }
