"""Anomaly Detection Engine — learn what normal looks like, flag what is not.

Pure NumPy, offline, deterministic. Isolation Forest + MLP autoencoder +
LSTM sequence autoencoder, one common ``fit / score / explain`` interface,
percentile thresholds from normal validation data, persistence + alert
grouping, reviewer-feedback retuning, and adapters for logs, sensors and
prices. See README.md for the measured benchmark.
"""

from .engine import AnomalyEngine
from .ensemble import EnsembleDetector
from .forest import IsolationForest
from .autoencoder import Autoencoder
from .lstm_ae import LSTMAutoencoder

__all__ = [
    "AnomalyEngine",
    "EnsembleDetector",
    "IsolationForest",
    "Autoencoder",
    "LSTMAutoencoder",
]
__version__ = "1.0.0"
