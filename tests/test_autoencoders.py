import numpy as np
import pytest

from anomalydetect.autoencoder import Autoencoder
from anomalydetect.features import window_features, window_labels, window_series
from anomalydetect.lstm_ae import LSTMAutoencoder
from anomalydetect.data import generate_normal, inject_anomalies


def _train_test():
    train = generate_normal(300, seed=11)
    clean = generate_normal(200, seed=12)
    test, labels, _ = inject_anomalies(clean, seed=13)
    return train, test, labels


def test_autoencoder_rebuilds_normal_better_than_anomalies():
    train, test, labels = _train_test()
    ae = Autoencoder(seed=2).fit(train, epochs=40)
    err = ae.reconstruction_error(test)
    assert err.shape == (200,)
    assert err[labels == 1].mean() > err[labels == 0].mean()
    feats = ae.feature_error(test)
    assert feats.shape == (200, 6)
    assert (feats >= 0).all()


def test_autoencoder_unfitted_raises():
    with pytest.raises(ValueError):
        Autoencoder().reconstruction_error(np.zeros((2, 6)))


def test_lstm_reconstructs_windows_and_flags_anomalies():
    from anomalydetect.scaling import StandardScaler

    train, test, labels = _train_test()
    win_tr = window_series(train, window=20, stride=10)
    win_te = window_series(test, window=20, stride=10)
    win_labels = window_labels(labels, window=20, stride=10)
    # The LSTM trains on scaled windows (raw sensor units are ~60-100, so
    # an unscaled error would be dominated by scale, not by anomalies).
    scaler = StandardScaler()
    scaler.fit(win_tr.reshape(-1, win_tr.shape[2]))
    win_tr = scaler.transform(win_tr.reshape(-1, win_tr.shape[2])).reshape(win_tr.shape)
    win_te = scaler.transform(win_te.reshape(-1, win_te.shape[2])).reshape(win_te.shape)
    ae = LSTMAutoencoder(seed=2).fit(win_tr, epochs=12)
    recon = ae.reconstruct(win_te)
    assert recon.shape == win_te.shape
    err = ae.reconstruction_error(win_te)
    assert err.shape == (len(win_te),)
    assert err[win_labels == 1].mean() > err[win_labels == 0].mean()


def test_windowing_shapes_and_labels():
    X = np.arange(100 * 6, dtype=float).reshape(100, 6)
    windows = window_series(X, window=20, stride=10)
    assert windows.shape == (9, 20, 6)
    feats = window_features(windows)
    assert feats.shape == (9, 36)
    labels = np.zeros(100, dtype=int)
    labels[50:55] = 1
    wl = window_labels(labels, window=20, stride=10)
    assert wl[4] == 1 and wl[0] == 0  # window starting at 40 covers 50-54
    with pytest.raises(ValueError):
        window_series(np.zeros((5,)))
