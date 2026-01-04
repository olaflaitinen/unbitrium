"""Quantity Skew Partitioner implementation.

Creates imbalanced dataset sizes across clients.

Author: Olaf Yunus Laitinen Imanov <oyli@dtu.dk>
License: EUPL-1.2
"""

from __future__ import annotations

from typing import Any

import numpy as np

from unbitrium.partitioning.base import Partitioner


class QuantitySkewPartitioner(Partitioner):
    """Quantity skew partitioner.

    Distributes samples with power-law imbalance in dataset sizes,
    creating quantity heterogeneity across clients.

    Args:
        num_clients: Number of clients.
        gamma: Power-law exponent (higher = more imbalanced).
        seed: Random seed.

    Example:
        >>> partitioner = QuantitySkewPartitioner(num_clients=10, gamma=1.5)
        >>> client_indices = partitioner.partition(labels)
    """

    def __init__(
        self,
        num_clients: int,
        gamma: float = 1.0,
        seed: int = 42,
    ) -> None:
        """Initialize quantity skew partitioner.

        Args:
            num_clients: Number of clients.
            gamma: Power-law exponent.
            seed: Random seed.
        """
        super().__init__(num_clients, seed)
        self.gamma = gamma

    def partition(self, labels: np.ndarray | Any) -> dict[int, list[int]]:
        """Partition with quantity skew.

        Args:
            labels: Array of class labels (used for sample count).

        Returns:
            Dictionary mapping client IDs to sample indices.
        """
        labels = self._get_targets(labels)
        num_samples = len(labels)
        rng = np.random.default_rng(self.seed)

        # Generate power-law proportions
        ranks = np.arange(1, self.num_clients + 1, dtype=np.float64)
        proportions = 1.0 / (ranks ** self.gamma)
        proportions = proportions / proportions.sum()

        # Convert to sample counts
        counts = (proportions * num_samples).astype(int)
        remainder = num_samples - counts.sum()
        for i in range(remainder):
            counts[i % self.num_clients] += 1

        # Shuffle all indices
        all_indices = np.arange(num_samples)
        rng.shuffle(all_indices)

        # Assign to clients
        client_indices: dict[int, list[int]] = {}
        curr = 0
        for client_id in range(self.num_clients):
            count = counts[client_id]
            client_indices[client_id] = all_indices[curr : curr + count].tolist()
            curr += count

        return client_indices
