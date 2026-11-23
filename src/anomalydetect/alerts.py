"""Persistence and alert grouping — the blueprint's false-alarm controls.

A single anomalous window is often noise. The persistence rule only raises
an alert after ``n`` consecutive anomalous windows, and grouping merges
alerts that fall within ``gap_minutes`` into one incident, so one fault
does not page anyone five times.
"""

from __future__ import annotations

import numpy as np


def apply_persistence(flags: np.ndarray, n: int = 2) -> np.ndarray:
    """Keep a flag only if it is part of a run of at least ``n`` True values."""
    flags = np.asarray(flags, dtype=bool)
    if n <= 1:
        return flags.copy()
    out = np.zeros_like(flags)
    run = 0
    for i, f in enumerate(flags):
        run = run + 1 if f else 0
        if run >= n:
            out[i - n + 1:i + 1] = True
    return out


def group_alerts(times_minutes: np.ndarray, flags: np.ndarray,
                 gap_minutes: float = 10.0) -> list[dict]:
    """Merge flagged times within ``gap_minutes`` into incidents.

    Each incident is ``{"start", "end", "count", "peak_index"}`` in the same
    index space as the inputs.
    """
    times = np.asarray(times_minutes, dtype=float)
    flags = np.asarray(flags, dtype=bool)
    idx = np.flatnonzero(flags)
    if len(idx) == 0:
        return []
    incidents: list[dict] = []
    start = prev = int(idx[0])
    count = 1
    for j in idx[1:]:
        j = int(j)
        if times[j] - times[prev] <= gap_minutes:
            prev, count = j, count + 1
        else:
            incidents.append({"start": start, "end": prev, "count": count,
                              "peak_index": start})
            start = prev = j
            count = 1
    incidents.append({"start": start, "end": prev, "count": count,
                      "peak_index": start})
    return incidents
