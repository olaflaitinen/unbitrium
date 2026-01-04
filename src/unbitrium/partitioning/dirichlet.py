"""Dirichlet Label Skew Partitioner.

Implements Dirichlet-multinomial sampling for creating non-IID client
data distributions with controlled label skew.

Mathematical formulation:

$$
p_k \\sim \\text{Dirichlet}(\\alpha \\mathbf{1})
$$

where $\\alpha$ controls heterogeneity (lower = more skewed).

Author: Olaf Yunus Laitinen Imanov <oyli@dtu.dk>
License: EUPL-1.2
"""

from __future__ import annotations

from typing import Any

import numpy as np

from unbitrium.partitioning.base import Partitioner


class DirichletPartitioner(Partitioner):
    """Dirichlet-based label skew partitioner.

    Distributes data across clients using Dirichlet-sampled proportions
    for each class, creating controlled heterogeneous distributions.

    Args:
        num_clients: Number of clients to partition data across.
        alpha: Dirichlet concentration parameter. Lower values create
            more heterogeneous distributions.
        seed: Random seed for reproducibility.

    Example:
        >>> partitioner = DirichletPartitioner(num_clients=10, alpha=0.5)
        >>> client_indices = partitioner.partition(labels)
    """

    def __init__(
        self,
        num_clients: int,
        alpha: float = 0.5,
        seed: int = 42,
    ) -> None:
        """Initialize Dirichlet partitioner.

        Args:
            num_clients: Number of clients.
            alpha: Dirichlet concentration parameter.
            seed: Random seed.
        """
        super().__init__(num_clients, seed)
        self.alpha = alpha

    def partition(self, labels: np.ndarray | Any) -> dict[int, list[int]]:
        """Partition dataset indices by Dirichlet-sampled label distributions.

        Args:
            labels: Array of class labels for each sample.

        Returns:
            Dictionary mapping client IDs to lists of sample indices.
        """
        if hasattr(labels, "numpy"):
            labels = labels.numpy()
        labels = np.asarray(labels)

        num_classes = len(np.unique(labels))
        num_samples = len(labels)

        # Get indices per class
        class_indices = [np.where(labels == c)[0] for c in range(num_classes)]

        client_indices: dict[int, list[int]] = {i: [] for i in range(self.num_clients)}
        rng = np.random.default_rng(self.seed)

        # Distribute each class according to Dirichlet-sampled proportions
        for c in range(num_classes):
            indices = class_indices[c].copy()
            num_class_samples = len(indices)

            if num_class_samples == 0:
                continue

            # Sample proportions from Dirichlet distribution
            proportions = rng.dirichlet(np.repeat(self.alpha, self.num_clients))

            # Convert proportions to sample counts
            counts = (proportions * num_class_samples).astype(int)
            # Handle rounding errors
            remainder = num_class_samples - counts.sum()
            for i in range(remainder):
                counts[i % self.num_clients] += 1

            # Shuffle and assign indices
            rng.shuffle(indices)
            current = 0
            for client_id in range(self.num_clients):
                count = counts[client_id]
                assigned = indices[current : current + count].tolist()
                client_indices[client_id].extend(assigned)
                current += count

        return client_indices


# Alias for backward compatibility
DirichletLabelSkew = DirichletPartitioner
