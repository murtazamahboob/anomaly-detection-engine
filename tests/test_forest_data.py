import numpy as np
import pytest

from anomalydetect.data import generate_normal, inject_anomalies, make_dataset
from anomalydetect.forest import IsolationForest, _c_factor
from anomalydetect.scaling import StandardScaler


def test_c_factor_known_values():
    assert _c_factor(1) == 0.0
    assert _c_factor(2) == 1.0
    assert _c_factor(256) == pytest.approx(10.24, abs=0.05)


def test_scaler_roundtrip_and_constant_feature():
    X = np.array([[1.0, 5.0], [2.0, 5.0], [3.0, 5.0]])
    scaler = StandardScaler()
    scaled = scaler.fit_transform(X)
    assert scaled[:, 1].tolist() == [0.0, 0.0, 0.0]  # constant -> scale 1
    np.testing.assert_allclose(scaler.inverse_transform(scaled), X)
    with pytest.raises(ValueError):
        StandardScaler().transform(X)


def test_data_is_deterministic_and_labelled():
    a = generate_normal(200, seed=3)
    b = generate_normal(200, seed=3)
    np.testing.assert_array_equal(a, b)
    assert a.shape == (200, 6)
    X, labels, segments = inject_anomalies(a, seed=3)
    assert len(segments) == 4
    assert {s["kind"] for s in segments} == {
        "spike", "level_shift", "drift", "dropout"}
    for s in segments:
        assert labels[s["start"]:s["end"]].all()
    assert labels.sum() == sum(s["end"] - s["start"] for s in segments)


def test_forest_scores_anomalies_higher():
    data = make_dataset(n_train=300, n_test=300)
    forest = IsolationForest(n_estimators=30, seed=1)
    forest.fit(data["train"])
    scores = forest.anomaly_score(data["test"])
    assert scores.shape == (300,)
    assert ((scores > 0) & (scores <= 1)).all()
    assert scores[data["test_labels"] == 1].mean() > \
        scores[data["test_labels"] == 0].mean()


def test_forest_is_deterministic():
    X = generate_normal(200, seed=5)
    f1 = IsolationForest(n_estimators=20, seed=9).fit(X)
    f2 = IsolationForest(n_estimators=20, seed=9).fit(X)
    np.testing.assert_array_equal(f1.anomaly_score(X), f2.anomaly_score(X))


def test_forest_rejects_bad_input():
    with pytest.raises(ValueError):
        IsolationForest().fit(np.ones((1, 3)))
    with pytest.raises(ValueError):
        IsolationForest().anomaly_score(np.ones((3, 3)))
