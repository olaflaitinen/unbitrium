"""Feature Shift Partitioner implementation.

Partitions data based on input feature distributions.

Author: Olaf Yunus Laitinen Imanov <oyli@dtu.dk>
License: EUPL-1.2
"""

from __future__ import annotations

from typing import Any

import numpy as np

from unbitrium.partitioning.base import Partitioner


class FeatureShiftPartitioner(Partitioner):
    """Feature shift partitioner.

    Creates heterogeneity based on input feature distributions rather
    than label distributions, simulating covariate shift.

    Args:
        num_clients: Number of clients.
        num_clusters: Number of feature clusters.
        seed: Random seed.

    Example:
        >>> partitioner = FeatureShiftPartitioner(num_clients=10, num_clusters=5)
        >>> client_indices = partitioner.partition(features)
    """

    def __init__(
        self,
        num_clients: int,
        num_clusters: int | None = None,
        seed: int = 42,
    ) -> None:
        """Initialize feature shift partitioner.

        Args:
            num_clients: Number of clients.
            num_clusters: Number of feature clusters (default: num_clients).
            seed: Random seed.
        """
        super().__init__(num_clients, seed)
        self.num_clusters = num_clusters or num_clients

    def partition(self, features: np.ndarray | Any) -> dict[int, list[int]]:
        """Partition based on feature clustering.

        Args:
            features: 2D array of input features.

        Returns:
            Dictionary mapping client IDs to sample indices.
        """
        if hasattr(features, "numpy"):
            features = features.numpy()
        features = np.asarray(features)

        if features.ndim == 1:
            features = features.reshape(-1, 1)

        num_samples = len(features)
        rng = np.random.default_rng(self.seed)

        # Simple k-means clustering
        cluster_labels = self._kmeans(features, self.num_clusters, rng)

        # Map clusters to clients (round-robin if more clusters than clients)
        client_indices: dict[int, list[int]] = {i: [] for i in range(self.num_clients)}

        for idx, cluster in enumerate(cluster_labels):
            client_id = cluster % self.num_clients
            client_indices[client_id].append(idx)

        return client_indices

    def _kmeans(
        self,
        features: np.ndarray,
        k: int,
        rng: np.random.Generator,
        max_iters: int = 100,
    ) -> np.ndarray:
        """Simple k-means clustering.

        Args:
            features: Feature matrix.
            k: Number of clusters.
            rng: Random generator.
            max_iters: Maximum iterations.

        Returns:
            Cluster labels for each sample.
        """
        num_samples = len(features)

        # Initialize centroids randomly
        centroid_indices = rng.choice(num_samples, size=k, replace=False)
        centroids = features[centroid_indices].copy()

        labels = np.zeros(num_samples, dtype=int)

        for _ in range(max_iters):
            # Assign to nearest centroid
            new_labels = np.zeros(num_samples, dtype=int)
            for i in range(num_samples):
                distances = np.linalg.norm(features[i] - centroids, axis=1)
                new_labels[i] = np.argmin(distances)

            # Check convergence
            if np.array_equal(labels, new_labels):
                break
            labels = new_labels

            # Update centroids
            for c in range(k):
                mask = labels == c
                if mask.any():
                    centroids[c] = features[mask].mean(axis=0)

        return labels
