"""Adapters: turn logs, sensor streams and price ticks into feature rows.

The engine scores feature rows; these adapters build those rows from the
three domains the blueprint names. Each adapter is a pure function, and each
one documents exactly which columns it produces, so a scored row can always
be traced back to the raw record that made it.
"""

from __future__ import annotations

import math
import re

import numpy as np

_LOG_RE = re.compile(
    r"(?P<minute>\d+)\s+(?P<level>INFO|WARN|ERROR)\s+latency=(?P<latency>\d+)"
)


def log_lines_to_features(lines: list[str]) -> np.ndarray:
    """One row per minute: [requests, error_rate, mean_latency, p95_latency].

    Lines look like ``"12 INFO latency=84"`` — minute, level, latency in ms.
    Minutes with no lines still get a row (zeros), so a silent minute is
    visible as a dropout rather than vanishing.
    """
    by_minute: dict[int, list[tuple[str, int]]] = {}
    for line in lines:
        m = _LOG_RE.search(line)
        if not m:
            continue
        by_minute.setdefault(int(m.group("minute")), []).append(
            (m.group("level"), int(m.group("latency")))
        )
    if not by_minute:
        return np.zeros((0, 4))
    rows = []
    for minute in range(min(by_minute), max(by_minute) + 1):
        entries = by_minute.get(minute, [])
        latencies = [lat for _, lat in entries]
        rows.append([
            float(len(entries)),
            float(sum(1 for lvl, _ in entries if lvl == "ERROR")
                  / max(len(entries), 1)),
            float(np.mean(latencies)) if latencies else 0.0,
            float(np.percentile(latencies, 95)) if latencies else 0.0,
        ])
    return np.array(rows)


def price_ticks_to_features(ticks: list[dict]) -> np.ndarray:
    """One row per tick: [spread, range_vs_atr, tick_rate, return].

    Each tick is ``{"bid", "ask", "price", "atr", "dt_seconds"}``. The
    blueprint's price features are spread, range relative to ATR and tick
    rate; return is added because a jump is itself a signal.
    """
    rows = []
    prev_price = None
    for tick in ticks:
        spread = float(tick["ask"] - tick["bid"])
        atr = float(tick.get("atr") or 1.0)
        dt = float(tick.get("dt_seconds") or 1.0)
        price = float(tick["price"])
        ret = 0.0 if prev_price is None else (price - prev_price) / prev_price
        rows.append([spread, spread / atr, 1.0 / max(dt, 1e-6), ret])
        prev_price = price
    return np.array(rows) if rows else np.zeros((0, 4))


def sensor_rows_to_features(rows: np.ndarray) -> np.ndarray:
    """Sensor adapter: pass the raw channels through, validating shape."""
    X = np.asarray(rows, dtype=float)
    if X.ndim != 2 or X.shape[0] == 0 or X.shape[1] == 0:
        raise ValueError("sensor rows must be a non-empty 2-D array")
    if not np.isfinite(X).all():
        # A NaN sensor reading is a dropout, not a number: mark it 0 and let
        # the detectors see the flat line. The raw value is never invented.
        X = np.where(np.isfinite(X), X, 0.0)
    return X


def feature_names_for(kind: str, n: int) -> list[str]:
    if kind == "logs":
        return ["requests_per_min", "error_rate", "mean_latency_ms", "p95_latency_ms"]
    if kind == "prices":
        return ["spread", "spread_over_atr", "tick_rate", "return"]
    return [f"sensor_{i}" for i in range(n)]
