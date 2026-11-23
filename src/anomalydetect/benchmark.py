"""Injected-anomaly benchmark: precision, recall, F1, PR-AUC, detection
delay, false-alarm rate — strict and point-adjusted, as the blueprint asks.

Point-adjusted recall gives credit if *any* window inside an anomalous
segment is flagged; strict recall counts every anomalous window. Reporting
both is the honest choice: point-adjusted numbers are always higher, and a
benchmark that only shows them is flattering itself.
"""

from __future__ import annotations

import numpy as np

from .data import FEATURE_NAMES, make_dataset
from .engine import AnomalyEngine
from .features import window_features, window_labels, window_series


def precision_recall_f1(scores: np.ndarray, labels: np.ndarray,
                        threshold: float) -> dict:
    pred = scores > threshold
    labels = np.asarray(labels, dtype=int)
    tp = int((pred & (labels == 1)).sum())
    fp = int((pred & (labels == 0)).sum())
    fn = int(((~pred) & (labels == 1)).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {"precision": precision, "recall": recall, "f1": f1,
            "tp": tp, "fp": fp, "fn": fn}


def pr_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Average precision (area under the precision-recall curve)."""
    order = np.argsort(-np.asarray(scores, dtype=float))
    labels = np.asarray(labels, dtype=int)[order]
    total_pos = max(int(labels.sum()), 1)
    tp = np.cumsum(labels == 1)
    fp = np.cumsum(labels == 0)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / total_pos
    # Step-integrate precision over recall.
    return float(np.sum(np.diff(np.concatenate([[0.0], recall])) * precision))


def point_adjusted_recall(scores: np.ndarray, labels: np.ndarray,
                          threshold: float, segments: list[dict]) -> float:
    """Share of anomalous *segments* with at least one flagged window."""
    if not segments:
        return 0.0
    pred = np.asarray(scores) > threshold
    hits = 0
    for seg in segments:
        if pred[seg["start"]:seg["end"]].any():
            hits += 1
    return hits / len(segments)


def detection_delay_windows(scores: np.ndarray, labels: np.ndarray,
                            threshold: float, segments: list[dict]) -> float | None:
    """Mean windows from a segment's start to its first flag (None if none)."""
    pred = np.asarray(scores) > threshold
    delays = []
    for seg in segments:
        flagged = np.flatnonzero(pred[seg["start"]:seg["end"]])
        if len(flagged):
            delays.append(int(flagged[0]))
    return float(np.mean(delays)) if delays else None


def run_benchmark(seed: int = 42, window: int = 20, stride: int = 10) -> dict:
    """Train every detector on normal data, score the injected test set."""
    data = make_dataset(seed=seed)
    train, test = data["train"], data["test"]
    labels, segments = data["test_labels"], data["segments"]
    # Window-level ground truth for the LSTM (it scores windows, not rows).
    win_labels = window_labels(labels, window, stride)
    # Window segments are the runs of anomalous windows — deriving them from
    # the labels, not from arithmetic on row indices, so a short segment
    # (e.g. a 3-row spike) is never mapped to an empty window range.
    win_segments: list[dict] = []
    run_start = None
    for i, lab in enumerate(win_labels):
        if lab == 1 and run_start is None:
            run_start = i
        if lab == 0 and run_start is not None:
            win_segments.append({"start": run_start, "end": i})
            run_start = None
    if run_start is not None:
        win_segments.append({"start": run_start, "end": len(win_labels)})
    results: dict = {"seed": seed, "window": window, "stride": stride,
                     "segments": segments, "methods": {}}
    # Row-level detectors: forest, autoencoder, ensemble.
    for name in ("forest", "autoencoder", "ensemble"):
        engine = AnomalyEngine(detector=name, seed=seed)
        engine.fit(train, feature_names=FEATURE_NAMES)
        scores = engine.score(test)
        metrics = precision_recall_f1(scores, labels, engine.threshold_)
        metrics["pr_auc"] = pr_auc(scores, labels)
        metrics["point_adjusted_recall"] = point_adjusted_recall(
            scores, labels, engine.threshold_, segments)
        metrics["detection_delay_rows"] = detection_delay_windows(
            scores, labels, engine.threshold_, segments)
        metrics["false_alarm_rate"] = float(
            ((scores > engine.threshold_) & (labels == 0)).sum()
            / max((labels == 0).sum(), 1))
        metrics["threshold"] = engine.threshold_
        results["methods"][name] = metrics
    # Window-level detectors: LSTM, and the ensemble on window features.
    windows_test = window_series(test, window, stride)
    for name in ("lstm",):
        engine = AnomalyEngine(detector=name, seed=seed)
        engine.fit(train, feature_names=FEATURE_NAMES)
        scores = engine.score(test)
        metrics = precision_recall_f1(scores, win_labels, engine.threshold_)
        metrics["pr_auc"] = pr_auc(scores, win_labels)
        metrics["point_adjusted_recall"] = point_adjusted_recall(
            scores, win_labels, engine.threshold_, win_segments)
        metrics["detection_delay_rows"] = detection_delay_windows(
            scores, win_labels, engine.threshold_, win_segments)
        metrics["false_alarm_rate"] = float(
            ((scores > engine.threshold_) & (win_labels == 0)).sum()
            / max((win_labels == 0).sum(), 1))
        metrics["threshold"] = engine.threshold_
        results["methods"][name] = metrics
    # Window-feature detectors: the same row detectors, but on per-window
    # aggregates (mean/std/min/max/range/slope). A level shift that stays
    # inside a channel's normal range is invisible row-by-row, but the
    # window mean moves — this is the comparison that shows it.
    train_w = window_features(window_series(train, window, stride))
    test_w = window_features(windows_test)
    win_names = [f"{agg}_{f}" for agg in
                 ("mean", "std", "min", "max", "range", "slope")
                 for f in FEATURE_NAMES]
    for name in ("forest", "autoencoder", "ensemble"):
        engine = AnomalyEngine(detector=name, seed=seed)
        engine.fit(train_w, feature_names=win_names)
        scores = engine.score(test_w)
        metrics = precision_recall_f1(scores, win_labels, engine.threshold_)
        metrics["pr_auc"] = pr_auc(scores, win_labels)
        metrics["point_adjusted_recall"] = point_adjusted_recall(
            scores, win_labels, engine.threshold_, win_segments)
        metrics["detection_delay_rows"] = detection_delay_windows(
            scores, win_labels, engine.threshold_, win_segments)
        metrics["false_alarm_rate"] = float(
            ((scores > engine.threshold_) & (win_labels == 0)).sum()
            / max((win_labels == 0).sum(), 1))
        metrics["threshold"] = engine.threshold_
        results["methods"][f"{name}_windows"] = metrics
    return results


def method_table(results: dict) -> str:
    lines = ["| Method | Precision | Recall | F1 | PR-AUC | Pt-adj recall | "
             "False alarms | Delay |",
             "|---|---|---|---|---|---|---|---|"]
    for name, m in results["methods"].items():
        delay = m["detection_delay_rows"]
        lines.append(
            f"| {name} | {m['precision']:.3f} | {m['recall']:.3f} | "
            f"{m['f1']:.3f} | {m['pr_auc']:.3f} | "
            f"{m['point_adjusted_recall']:.3f} | {m['false_alarm_rate']:.3f} | "
            f"{delay if delay is None else round(delay, 1)} |")
    return "\n".join(lines)
