"""Distribution metrics for federated learning.

Provides functions to measure data distribution properties.

Author: Olaf Yunus Laitinen Imanov <oyli@dtu.dk>
License: EUPL-1.2
"""

from __future__ import annotations

from typing import Any

import numpy as np


def compute_distribution_metrics(
    labels: np.ndarray | Any,
    client_indices: dict[int, list[int]],
) -> dict[str, float]:
    """Compute comprehensive distribution metrics.

    Args:
        labels: Array of class labels.
        client_indices: Mapping from client ID to sample indices.

    Returns:
        Dictionary of distribution metrics.
    """
    if hasattr(labels, "numpy"):
        labels = labels.numpy()
    labels = np.asarray(labels)

    num_classes = len(np.unique(labels))
    num_clients = len(client_indices)

    # Per-client sizes
    sizes = [len(indices) for indices in client_indices.values()]

    # Global distribution
    global_counts = np.bincount(labels, minlength=num_classes)
    global_dist = global_counts / global_counts.sum()

    # Client distributions
    client_dists = []
    for indices in client_indices.values():
        if len(indices) > 0:
            client_labels = labels[indices]
            counts = np.bincount(client_labels, minlength=num_classes)
            client_dists.append(counts / counts.sum())

    # Coefficient of variation for sizes
    size_cv = np.std(sizes) / np.mean(sizes) if np.mean(sizes) > 0 else 0.0

    # Average L1 distance from global
    l1_distances = [np.sum(np.abs(d - global_dist)) for d in client_dists]
    avg_l1 = np.mean(l1_distances) if l1_distances else 0.0

    # Gini coefficient for class balance
    ginis = []
    for dist in client_dists:
        sorted_dist = np.sort(dist)
        n = len(sorted_dist)
        cumsum = np.cumsum(sorted_dist)
        gini = (2 * np.sum((np.arange(1, n + 1) * sorted_dist)) / (n * cumsum[-1])) - (n + 1) / n
        ginis.append(gini)
    avg_gini = np.mean(ginis) if ginis else 0.0

    return {
        "num_clients": float(num_clients),
        "num_classes": float(num_classes),
        "total_samples": float(len(labels)),
        "min_client_size": float(min(sizes)),
        "max_client_size": float(max(sizes)),
        "avg_client_size": float(np.mean(sizes)),
        "size_cv": float(size_cv),
        "avg_l1_distance": float(avg_l1),
        "avg_gini": float(avg_gini),
    }
