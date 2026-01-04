"""Representation metrics for federated learning.

Provides feature representation and embedding metrics.

Author: Olaf Yunus Laitinen Imanov <oyli@dtu.dk>
License: EUPL-1.2
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch


def compute_representation_similarity(
    representations_1: np.ndarray | torch.Tensor,
    representations_2: np.ndarray | torch.Tensor,
) -> float:
    """Compute cosine similarity between representation sets.

    Args:
        representations_1: First set of representations.
        representations_2: Second set of representations.

    Returns:
        Average cosine similarity.
    """
    if isinstance(representations_1, torch.Tensor):
        representations_1 = representations_1.numpy()
    if isinstance(representations_2, torch.Tensor):
        representations_2 = representations_2.numpy()

    # Compute mean representations
    mean_1 = np.mean(representations_1, axis=0)
    mean_2 = np.mean(representations_2, axis=0)

    # Cosine similarity
    norm_1 = np.linalg.norm(mean_1)
    norm_2 = np.linalg.norm(mean_2)

    if norm_1 < 1e-10 or norm_2 < 1e-10:
        return 0.0

    return float(np.dot(mean_1, mean_2) / (norm_1 * norm_2))


def compute_representation_variance(
    representations: np.ndarray | torch.Tensor,
) -> float:
    """Compute variance of representations.

    Args:
        representations: Feature representations.

    Returns:
        Mean variance across dimensions.
    """
    if isinstance(representations, torch.Tensor):
        representations = representations.numpy()

    return float(np.mean(np.var(representations, axis=0)))


def compute_cka_similarity(
    X: np.ndarray | torch.Tensor,
    Y: np.ndarray | torch.Tensor,
) -> float:
    """Compute Centered Kernel Alignment (CKA) similarity.

    Args:
        X: First representation matrix (n x d1).
        Y: Second representation matrix (n x d2).

    Returns:
        CKA similarity score.
    """
    if isinstance(X, torch.Tensor):
        X = X.numpy()
    if isinstance(Y, torch.Tensor):
        Y = Y.numpy()

    # Center the matrices
    X = X - np.mean(X, axis=0, keepdims=True)
    Y = Y - np.mean(Y, axis=0, keepdims=True)

    # Compute Gram matrices with linear kernel
    K_X = X @ X.T
    K_Y = Y @ Y.T

    # Compute HSIC
    def hsic(K, L):
        n = K.shape[0]
        H = np.eye(n) - np.ones((n, n)) / n
        return np.trace(K @ H @ L @ H) / ((n - 1) ** 2)

    hsic_xy = hsic(K_X, K_Y)
    hsic_xx = hsic(K_X, K_X)
    hsic_yy = hsic(K_Y, K_Y)

    denom = np.sqrt(hsic_xx * hsic_yy)
    if denom < 1e-10:
        return 0.0

    return float(hsic_xy / denom)
