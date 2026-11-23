import numpy as np
import pytest

from anomalydetect.benchmark import (method_table, point_adjusted_recall,
                                     pr_auc, precision_recall_f1)
from anomalydetect.data import FEATURE_NAMES, make_dataset
from anomalydetect.engine import AnomalyEngine
from anomalydetect.ensemble import EnsembleDetector


def test_metric_functions():
    scores = np.array([0.9, 0.8, 0.2, 0.1])
    labels = np.array([1, 1, 0, 0])
    m = precision_recall_f1(scores, labels, 0.5)
    assert m == {"precision": 1.0, "recall": 1.0, "f1": 1.0,
                 "tp": 2, "fp": 0, "fn": 0}
    assert pr_auc(scores, labels) == pytest.approx(1.0)
    assert pr_auc(np.array([0.1, 0.2, 0.8, 0.9]), labels) < 0.6
    segments = [{"start": 0, "end": 2}]
    assert point_adjusted_recall(scores, labels, 0.5, segments) == 1.0
    assert point_adjusted_recall(scores, labels, 0.5, []) == 0.0


def test_engine_all_detectors_flag_injected_segments():
    data = make_dataset(n_train=300, n_test=300)
    for detector in ("forest", "autoencoder", "ensemble"):
        engine = AnomalyEngine(detector=detector, seed=1)
        engine.fit(data["train"], feature_names=FEATURE_NAMES)
        assert engine.threshold_ is not None
        scores = engine.score(data["test"])
        assert scores.shape == (300,)
        flags = engine.predict(data["test"])
        assert flags.dtype == bool
        # At least one injected segment is caught by every detector.
        caught = [s for s in data["segments"]
                  if flags[s["start"]:s["end"]].any()]
        assert len(caught) >= 1, detector
        expl = engine.explain(data["test"][:5])
        assert expl.shape == (5, 6)
        np.testing.assert_allclose(expl.sum(axis=1), 1.0)
        tops = engine.top_features(data["test"][:2], k=2)
        assert tops[0][0][0] in FEATURE_NAMES


def test_engine_rejects_unknown_detector_and_unfitted_use():
    with pytest.raises(ValueError):
        AnomalyEngine(detector="nope")
    with pytest.raises(ValueError):
        AnomalyEngine().score(np.zeros((2, 6)))


def test_ensemble_scores_ranked_and_votes_bounded():
    data = make_dataset(n_train=300, n_test=200)
    ens = EnsembleDetector(n_estimators=20, seed=1)
    ens.fit(data["train"])
    scores = ens.score(data["test"])
    assert ((scores >= 0) & (scores <= 1)).all()
    members = ens.member_scores(data["test"])
    assert set(members) == {"isolation_forest", "autoencoder"}
    votes = ens.votes(data["test"], 0.6, 0.0)
    assert set(np.unique(votes)).issubset({0, 1, 2})


def test_method_table_lists_methods():
    results = {"methods": {"forest": {
        "precision": 0.8, "recall": 0.7, "f1": 0.74, "pr_auc": 0.75,
        "point_adjusted_recall": 1.0, "false_alarm_rate": 0.01,
        "detection_delay_rows": 2.0}}}
    table = method_table(results)
    assert "forest" in table and "0.800" in table
