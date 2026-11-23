"""MLP autoencoder from scratch (NumPy, Adam, early stopping).

Trained on normal data only, so it learns to rebuild normal structure and
rebuilds anomalies badly. The anomaly score is the per-sample mean squared
reconstruction error; the per-feature squared error is the explanation
heatmap (which channel the model could not rebuild).
"""

from __future__ import annotations

import numpy as np


def _relu(z: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, z)


def _relu_grad(z: np.ndarray) -> np.ndarray:
    return (z > 0).astype(float)


class Autoencoder:
    """d -> 32 -> 16 -> bottleneck -> 16 -> 32 -> d, ReLU hidden, linear out."""

    def __init__(self, bottleneck: int = 8, hidden: tuple[int, int] = (32, 16),
                 seed: int = 42):
        self.bottleneck = bottleneck
        self.hidden = hidden
        self.seed = seed
        self.weights_: list[np.ndarray] = []
        self.biases_: list[np.ndarray] = []
        self.n_features_: int = 0
        self.history_: dict = {"train_loss": [], "val_loss": []}

    def _init(self, d: int, rng: np.random.Generator) -> None:
        sizes = [d, *self.hidden, self.bottleneck, *reversed(self.hidden), d]
        self.weights_, self.biases_ = [], []
        for fan_in, fan_out in zip(sizes, sizes[1:]):
            limit = np.sqrt(6.0 / (fan_in + fan_out))
            self.weights_.append(rng.uniform(-limit, limit, (fan_in, fan_out)))
            self.biases_.append(np.zeros(fan_out))
        self.n_features_ = d

    def _forward(self, X: np.ndarray):
        acts = [X]
        pre = []
        a = X
        for i, (W, b) in enumerate(zip(self.weights_, self.biases_)):
            z = a @ W + b
            pre.append(z)
            a = z if i == len(self.weights_) - 1 else _relu(z)
            acts.append(a)
        return acts, pre

    def _backward(self, acts, pre, X: np.ndarray):
        n = len(X)
        grads_W = [None] * len(self.weights_)
        grads_b = [None] * len(self.biases_)
        delta = (acts[-1] - X) * (2.0 / (n * X.shape[1]))
        for i in reversed(range(len(self.weights_))):
            grads_W[i] = acts[i].T @ delta
            grads_b[i] = delta.sum(axis=0)
            if i > 0:
                delta = (delta @ self.weights_[i].T) * _relu_grad(pre[i - 1])
        return grads_W, grads_b

    def fit(self, X: np.ndarray, X_val: np.ndarray | None = None,
            epochs: int = 120, lr: float = 1e-3, batch_size: int = 64,
            patience: int = 15) -> "Autoencoder":
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be a 2-D array")
        rng = np.random.default_rng(self.seed)
        self._init(X.shape[1], rng)
        if X_val is None:
            split = int(0.85 * len(X))
            X_tr, X_val = X[:split], X[split:]
        else:
            X_tr = X
        m = [np.zeros_like(w) for w in self.weights_]
        v = [np.zeros_like(w) for w in self.weights_]
        mb = [np.zeros_like(b) for b in self.biases_]
        vb = [np.zeros_like(b) for b in self.biases_]
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        step = 0
        best_val, best_state, bad = float("inf"), None, 0
        for epoch in range(epochs):
            perm = rng.permutation(len(X_tr))
            for start in range(0, len(X_tr), batch_size):
                idx = perm[start:start + batch_size]
                batch = X_tr[idx]
                acts, pre = self._forward(batch)
                gW, gb = self._backward(acts, pre, batch)
                step += 1
                for i in range(len(self.weights_)):
                    m[i] = beta1 * m[i] + (1 - beta1) * gW[i]
                    v[i] = beta2 * v[i] + (1 - beta2) * gW[i] ** 2
                    mh = m[i] / (1 - beta1 ** step)
                    vh = v[i] / (1 - beta2 ** step)
                    self.weights_[i] -= lr * mh / (np.sqrt(vh) + eps)
                    mb[i] = beta1 * mb[i] + (1 - beta1) * gb[i]
                    vb[i] = beta2 * vb[i] + (1 - beta2) * gb[i] ** 2
                    mbh = mb[i] / (1 - beta1 ** step)
                    vbh = vb[i] / (1 - beta2 ** step)
                    self.biases_[i] -= lr * mbh / (np.sqrt(vbh) + eps)
            train_loss = float(np.mean(self.reconstruction_error(X_tr)))
            val_loss = float(np.mean(self.reconstruction_error(X_val)))
            self.history_["train_loss"].append(train_loss)
            self.history_["val_loss"].append(val_loss)
            if val_loss < best_val - 1e-6:
                best_val = val_loss
                best_state = ([w.copy() for w in self.weights_],
                              [b.copy() for b in self.biases_])
                bad = 0
            else:
                bad += 1
                if bad >= patience:
                    break
        if best_state is not None:
            self.weights_, self.biases_ = best_state
        return self

    def reconstruct(self, X: np.ndarray) -> np.ndarray:
        if not self.weights_:
            raise ValueError("autoencoder is not fitted")
        acts, _ = self._forward(np.asarray(X, dtype=float))
        return acts[-1]

    def reconstruction_error(self, X: np.ndarray) -> np.ndarray:
        """Per-sample MSE; higher = more anomalous."""
        X = np.asarray(X, dtype=float)
        return np.mean((self.reconstruct(X) - X) ** 2, axis=1)

    def feature_error(self, X: np.ndarray) -> np.ndarray:
        """Per-sample, per-feature squared error — the explanation heatmap."""
        X = np.asarray(X, dtype=float)
        return (self.reconstruct(X) - X) ** 2
