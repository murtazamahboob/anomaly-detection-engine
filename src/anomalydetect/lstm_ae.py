"""LSTM sequence autoencoder from scratch (NumPy, BPTT, Adam).

An LSTM encoder reads a window of ``T`` steps and compresses it into its
final hidden state; a small MLP decoder rebuilds the whole window from that
state. Trained on normal windows only, so a window with a spike, shift or
dropout rebuilds badly and its reconstruction error flags it. The blueprint
asks for an LSTM autoencoder for sequences — this is it, with the decoder as
a dense head on the LSTM's final state (documented honestly in the README).
"""

from __future__ import annotations

import numpy as np


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


class LSTMAutoencoder:
    """LSTM encoder (hidden 16) + MLP decoder, trained on normal windows."""

    def __init__(self, hidden: int = 16, seed: int = 42):
        self.hidden = hidden
        self.seed = seed
        self.params: dict[str, np.ndarray] = {}
        self.n_features_: int = 0
        self.window_: int = 0
        self.history_: dict = {"train_loss": [], "val_loss": []}

    # -- parameters -------------------------------------------------------
    def _init(self, d: int, T: int, rng: np.random.Generator) -> None:
        h = self.hidden
        scale = np.sqrt(1.0 / (d + h))
        self.params = {
            "W": rng.normal(0.0, scale, (d + h, 4 * h)),
            "b": np.zeros(4 * h),
            "W1": rng.normal(0.0, np.sqrt(1.0 / h), (h, 32)),
            "b1": np.zeros(32),
            "W2": rng.normal(0.0, np.sqrt(1.0 / 32), (32, T * d)),
            "b2": np.zeros(T * d),
        }
        self.n_features_, self.window_ = d, T

    # -- forward ----------------------------------------------------------
    def _encode(self, X: np.ndarray):
        """Return (final hidden state, per-step caches). X: (N, T, d)."""
        N, T, d = X.shape
        h = self.hidden
        W, b = self.params["W"], self.params["b"]
        h_prev = np.zeros((N, h))
        c_prev = np.zeros((N, h))
        caches = []
        for t in range(T):
            concat = np.concatenate([X[:, t, :], h_prev], axis=1)
            z = concat @ W + b
            i = _sigmoid(z[:, :h])
            f = _sigmoid(z[:, h:2 * h])
            o = _sigmoid(z[:, 2 * h:3 * h])
            g = np.tanh(z[:, 3 * h:])
            c = f * c_prev + i * g
            h_t = o * np.tanh(c)
            caches.append((concat, i, f, o, g, c_prev, c, h_t))
            h_prev, c_prev = h_t, c
        return h_prev, caches

    def _decode(self, h_last: np.ndarray):
        z1 = h_last @ self.params["W1"] + self.params["b1"]
        a1 = np.maximum(0.0, z1)
        out = a1 @ self.params["W2"] + self.params["b2"]
        return out, (z1, a1)

    def reconstruct(self, X: np.ndarray) -> np.ndarray:
        if not self.params:
            raise ValueError("LSTM autoencoder is not fitted")
        X = np.asarray(X, dtype=float)
        h_last, _ = self._encode(X)
        out, _ = self._decode(h_last)
        return out.reshape(X.shape)

    def reconstruction_error(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        return np.mean((self.reconstruct(X) - X) ** 2, axis=(1, 2))

    # -- training ---------------------------------------------------------
    def fit(self, X: np.ndarray, X_val: np.ndarray | None = None,
            epochs: int = 60, lr: float = 3e-3, batch_size: int = 32,
            patience: int = 10) -> "LSTMAutoencoder":
        X = np.asarray(X, dtype=float)
        if X.ndim != 3:
            raise ValueError("X must be (n_windows, T, n_features)")
        rng = np.random.default_rng(self.seed)
        self._init(X.shape[2], X.shape[1], rng)
        if X_val is None:
            split = int(0.85 * len(X))
            X_tr, X_val = X[:split], X[split:]
        else:
            X_tr = X
        keys = list(self.params)
        m = {k: np.zeros_like(self.params[k]) for k in keys}
        v = {k: np.zeros_like(self.params[k]) for k in keys}
        step, best_val, bad = 0, float("inf"), 0
        best_state = None
        for _ in range(epochs):
            perm = rng.permutation(len(X_tr))
            for start in range(0, len(X_tr), batch_size):
                batch = X_tr[perm[start:start + batch_size]]
                grads = self._gradients(batch)
                step += 1
                for k in keys:
                    m[k] = 0.9 * m[k] + 0.1 * grads[k]
                    v[k] = 0.999 * v[k] + 0.001 * grads[k] ** 2
                    mh = m[k] / (1 - 0.9 ** step)
                    vh = v[k] / (1 - 0.999 ** step)
                    self.params[k] -= lr * mh / (np.sqrt(vh) + 1e-8)
            train_loss = float(np.mean(self.reconstruction_error(X_tr)))
            val_loss = float(np.mean(self.reconstruction_error(X_val)))
            self.history_["train_loss"].append(train_loss)
            self.history_["val_loss"].append(val_loss)
            if val_loss < best_val - 1e-6:
                best_val, best_state, bad = val_loss, {k: p.copy() for k, p in self.params.items()}, 0
            else:
                bad += 1
                if bad >= patience:
                    break
        if best_state is not None:
            self.params = best_state
        return self

    def _gradients(self, X: np.ndarray) -> dict[str, np.ndarray]:
        N, T, d = X.shape
        h = self.hidden
        h_last, caches = self._encode(X)
        out, (z1, a1) = self._decode(h_last)
        target = X.reshape(N, -1)
        # Decoder backward (MSE, mean over N and T*d).
        delta_out = (out - target) * (2.0 / (N * T * d))
        grads = {
            "W2": a1.T @ delta_out,
            "b2": delta_out.sum(axis=0),
        }
        delta_a1 = delta_out @ self.params["W2"].T
        delta_z1 = delta_a1 * (z1 > 0)
        grads["W1"] = h_last.T @ delta_z1
        grads["b1"] = delta_z1.sum(axis=0)
        dh = delta_z1 @ self.params["W1"].T  # gradient into final hidden state
        # LSTM backward through time.
        dW = np.zeros_like(self.params["W"])
        db = np.zeros_like(self.params["b"])
        dc_next = np.zeros((N, h))
        W = self.params["W"]
        for t in reversed(range(T)):
            concat, i, f, o, g, c_prev, c, _ = caches[t]
            tanh_c = np.tanh(c)
            do = dh * tanh_c
            dc = dh * o * (1 - tanh_c ** 2) + dc_next
            di = dc * g
            df = dc * c_prev
            dg = dc * i
            dz = np.empty((N, 4 * h))
            dz[:, :h] = di * i * (1 - i)
            dz[:, h:2 * h] = df * f * (1 - f)
            dz[:, 2 * h:3 * h] = do * o * (1 - o)
            dz[:, 3 * h:] = dg * (1 - g ** 2)
            dW += concat.T @ dz
            db += dz.sum(axis=0)
            d_concat = dz @ W.T
            dh = d_concat[:, d:]
            dc_next = dc * f
        grads["W"], grads["b"] = dW, db
        return grads
