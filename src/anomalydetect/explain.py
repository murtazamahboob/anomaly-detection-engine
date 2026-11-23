"""Explanations: which feature, or which part of a window, caused the score.

Two honest, cheap explanations, one per detector family:

* Autoencoder: the per-feature squared reconstruction error. A channel the
  model cannot rebuild is the channel that is wrong.
* Isolation forest: for each tree, walk the point's path and count the splits
  on each feature, weighted by how early the split happened (early splits
  isolate more). Averaged over trees, features that split the point early
  are the ones that make it unusual.

The ensemble explanation is the rank-average of the two, so one method
cannot drown the other out. Every explanation is normalised to sum to 1, so
it reads as "share of blame".
"""

from __future__ import annotations

import numpy as np


def normalise_rows(M: np.ndarray) -> np.ndarray:
    M = np.asarray(M, dtype=float)
    totals = M.sum(axis=1, keepdims=True)
    totals = np.where(totals <= 0, 1.0, totals)
    return M / totals


def ae_feature_errors(reconstructed: np.ndarray, actual: np.ndarray) -> np.ndarray:
    return (np.asarray(reconstructed, dtype=float)
            - np.asarray(actual, dtype=float)) ** 2


def forest_feature_contributions(forest, X: np.ndarray) -> np.ndarray:
    """Per-sample, per-feature split credit along each point's tree paths."""
    X = np.asarray(X, dtype=float)
    n, d = X.shape
    out = np.zeros((n, d))
    for tree in forest.trees_:
        for i, x in enumerate(X):
            node = tree
            depth = 0
            while not node.is_leaf:
                # Early splits isolate more: weight 1 / (depth + 1).
                out[i, node.feature] += 1.0 / (depth + 1.0)
                node = node.left if x[node.feature] < node.split else node.right
                depth += 1
    if forest.trees_:
        out /= len(forest.trees_)
    return normalise_rows(out)


def combined_explanation(ae_errors: np.ndarray,
                         forest_contrib: np.ndarray) -> np.ndarray:
    """Rank-average of the two normalised explanations."""
    return normalise_rows(
        normalise_rows(ae_errors) + normalise_rows(forest_contrib)
    )


def top_features(explanation: np.ndarray, feature_names: list[str],
                 k: int = 3) -> list[list[tuple[str, float]]]:
    """Top-k blamed features per sample, as ``(name, share)`` pairs."""
    result = []
    for row in explanation:
        order = np.argsort(-row)[:k]
        result.append([(feature_names[j], float(row[j])) for j in order])
    return result
