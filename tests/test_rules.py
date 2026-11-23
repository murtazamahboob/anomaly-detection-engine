import numpy as np
import pytest

from anomalydetect.adapters import (feature_names_for, log_lines_to_features,
                                   price_ticks_to_features,
                                   sensor_rows_to_features)
from anomalydetect.alerts import apply_persistence, group_alerts
from anomalydetect.explain import (combined_explanation,
                                   forest_feature_contributions,
                                   normalise_rows, top_features)
from anomalydetect.feedback import false_alarm_rate, retune_threshold
from anomalydetect.forest import IsolationForest
from anomalydetect.threshold import (flag, mean_plus_std_threshold,
                                     percentile_threshold)


def test_threshold_helpers():
    scores = np.arange(100, dtype=float)
    # NumPy interpolates percentiles: the 99.5th of 0..99 is 98.505.
    assert percentile_threshold(scores, 99.5) == pytest.approx(98.505, abs=0.01)
    assert mean_plus_std_threshold(scores, 3.0) > scores.mean()
    assert flag(np.array([0.1, 0.9]), 0.5).tolist() == [False, True]
    with pytest.raises(ValueError):
        percentile_threshold(np.array([]))
    with pytest.raises(ValueError):
        percentile_threshold(scores, 100.0)


def test_persistence_needs_consecutive_flags():
    flags = np.array([True, False, True, True, False, True, True, True])
    out = apply_persistence(flags, n=2)
    assert out.tolist() == [False, False, True, True, False, True, True, True]
    assert apply_persistence(flags, n=1).tolist() == flags.tolist()


def test_grouping_merges_within_10_minutes():
    times = np.array([0.0, 2.0, 9.0, 25.0, 30.0])
    flags = np.array([True, True, True, True, False])
    incidents = group_alerts(times, flags, gap_minutes=10.0)
    assert len(incidents) == 2
    assert incidents[0]["start"] == 0 and incidents[0]["end"] == 2
    assert incidents[0]["count"] == 3
    assert group_alerts(times, np.zeros(5, dtype=bool)) == []


def test_feedback_retunes_threshold():
    scores = np.array([0.1, 0.2, 0.9, 0.95])
    labels = np.array([0, 0, 0, 1])  # 0.9 was a false alarm, 0.95 real
    new = retune_threshold(scores, labels, 0.5, target_false_alarm_rate=0.01)
    assert new == pytest.approx(0.9)  # rises to silence the false alarm
    # A missed real anomaly pulls the threshold down.
    labels2 = np.array([0, 0, 1, 1])
    new2 = retune_threshold(np.array([0.1, 0.2, 0.4, 0.95]), labels2, 0.5)
    assert new2 == pytest.approx(0.4)
    # Nothing reviewed: unchanged.
    assert retune_threshold(scores, np.array([-1, -1, -1, -1]), 0.5) == 0.5
    assert false_alarm_rate(scores, np.array([0, 0, 0, 1]), 0.5) == 0.5


def test_explanations_sum_to_one():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(20, 6))
    forest = IsolationForest(n_estimators=10, seed=1).fit(X)
    contrib = forest_feature_contributions(forest, X)
    assert contrib.shape == (20, 6)
    np.testing.assert_allclose(contrib.sum(axis=1), 1.0)
    combined = combined_explanation(np.abs(rng.normal(size=(20, 6))), contrib)
    np.testing.assert_allclose(combined.sum(axis=1), 1.0)
    tops = top_features(combined, [f"f{i}" for i in range(6)], k=2)
    assert len(tops) == 20 and len(tops[0]) == 2


def test_log_adapter_features():
    lines = ["0 INFO latency=80", "0 INFO latency=90", "0 ERROR latency=400",
             "1 INFO latency=70", "3 WARN latency=120"]
    feats = log_lines_to_features(lines)
    assert feats.shape == (4, 4)  # minutes 0..3, minute 2 is silent (zeros)
    assert feats[0, 0] == 3 and feats[0, 1] == pytest.approx(1 / 3)
    assert feats[2].tolist() == [0.0, 0.0, 0.0, 0.0]
    assert log_lines_to_features([]).shape == (0, 4)
    assert feature_names_for("logs", 4)[1] == "error_rate"


def test_price_adapter_and_sensor_nan():
    ticks = [{"bid": 100.0, "ask": 100.5, "price": 100.2, "atr": 1.0,
              "dt_seconds": 0.5},
             {"bid": 100.1, "ask": 100.6, "price": 100.4, "atr": 1.0,
              "dt_seconds": 0.5}]
    feats = price_ticks_to_features(ticks)
    assert feats.shape == (2, 4)
    assert feats[0, 0] == pytest.approx(0.5)
    assert feats[1, 2] == pytest.approx(2.0)  # tick rate 1/0.5s
    X = sensor_rows_to_features(np.array([[1.0, np.nan], [2.0, 3.0]]))
    assert X[0, 1] == 0.0
    with pytest.raises(ValueError):
        sensor_rows_to_features(np.zeros((0, 3)))
