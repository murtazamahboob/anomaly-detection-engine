"""Isolation Forest from scratch (NumPy only, no scikit-learn).

An anomaly is a point that is easy to isolate: random splits separate it in
few steps, so its average path length through the trees is short. The
anomaly score is the classic one from Liu et al. (2008):

    s(x) = 2 ** (-E[h(x)] / c(n)),   c(n) = 2*H(n-1) - 2*(n-1)/n

so s is near 1 for anomalies, near 0.5 for typical points, and below 0.5 for
dense clusters. Higher score = more anomalous, matching the autoencoder's
reconstruction error so the two can be ensembled.
"""

from __future__ import annotations

import numpy as np


def _harmonic(n: int) -> float:
    return float(np.sum(1.0 / np.arange(1, n + 1))) if n > 0 else 0.0


def _c_factor(n: int) -> float:
    """Average path length of an unsuccessful search in a BST of n nodes."""
    if n <= 1:
        return 0.0
    if n == 2:
        return 1.0
    return 2.0 * _harmonic(n - 1) - 2.0 * (n - 1) / n


class _Node:
    __slots__ = ("feature", "split", "left", "right", "size")

    def __init__(self, size: int, feature: int = -1, split: float = 0.0,
                 left=None, right=None):
        self.size = size
        self.feature = feature
        self.split = split
        self.left = left
        self.right = right

    @property
    def is_leaf(self) -> bool:
        return self.left is None


class IsolationForest:
    """Random-split isolation forest. ``fit`` sees normal data only."""

    def __init__(self, n_estimators: int = 300, max_samples: int = 256,
                 max_depth: int | None = None, seed: int = 42):
        if n_estimators < 1:
            raise ValueError("n_estimators must be >= 1")
        if max_samples < 2:
            raise ValueError("max_samples must be >= 2")
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.max_depth = max_depth
        self.seed = seed
        self.trees_: list[_Node] = []
        self.subsample_size_: int = 0

    # -- training ---------------------------------------------------------
    def fit(self, X: np.ndarray) -> "IsolationForest":
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[0] < 2:
            raise ValueError("X must be a 2-D array with at least 2 rows")
        rng = np.random.default_rng(self.seed)
        n = X.shape[0]
        self.subsample_size_ = min(self.max_samples, n)
        limit = self.max_depth or int(np.ceil(np.log2(self.subsample_size_)))
        self.trees_ = []
        for _ in range(self.n_estimators):
            idx = rng.choice(n, size=self.subsample_size_, replace=False)
            self.trees_.append(self._build(X[idx], depth=0, limit=limit, rng=rng))
        return self

    def _build(self, X: np.ndarray, depth: int, limit: int,
               rng: np.random.Generator) -> _Node:
        n, d = X.shape
        if n <= 1 or depth >= limit:
            return _Node(size=n)
        # Pick a feature that actually varies in this subsample.
        spreads = X.max(axis=0) - X.min(axis=0)
        candidates = np.flatnonzero(spreads > 0)
        if len(candidates) == 0:
            return _Node(size=n)
        feat = int(rng.choice(candidates))
        lo, hi = X[:, feat].min(), X[:, feat].max()
        split = float(rng.uniform(lo, hi))
        left_mask = X[:, feat] < split
        # A split that isolates nothing is wasted; fall back to a leaf.
        if left_mask.all() or (~left_mask).all():
            return _Node(size=n)
        return _Node(
            size=n, feature=feat, split=split,
            left=self._build(X[left_mask], depth + 1, limit, rng),
            right=self._build(X[~left_mask], depth + 1, limit, rng),
        )

    # -- scoring ----------------------------------------------------------
    def _path_length(self, x: np.ndarray, node: _Node, depth: int = 0) -> float:
        while not node.is_leaf:
            if x[node.feature] < node.split:
                node = node.left
            else:
                node = node.right
            depth += 1
        # Unfinished leaves carry the expected remaining path length.
        return depth + _c_factor(node.size)

    def anomaly_score(self, X: np.ndarray) -> np.ndarray:
        """Score in (0, 1]; higher = more anomalous."""
        if not self.trees_:
            raise ValueError("forest is not fitted")
        X = np.asarray(X, dtype=float)
        c = _c_factor(self.subsample_size_)
        if c == 0:
            return np.zeros(len(X))
        mean_depth = np.array(
            [
                np.mean([self._path_length(x, tree) for tree in self.trees_])
                for x in X
            ]
        )
        return 2.0 ** (-mean_depth / c)

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        return self.anomaly_score(X)
