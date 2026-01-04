"""MoDM Partitioner implementation.

Mixture-of-Dirichlet-Multinomials for multi-modal heterogeneity.

Author: Olaf Yunus Laitinen Imanov <oyli@dtu.dk>
License: EUPL-1.2
"""

from __future__ import annotations

from typing import Any

import numpy as np

from unbitrium.partitioning.base import Partitioner


class MoDMPartitioner(Partitioner):
    """Mixture-of-Dirichlet-Multinomials partitioner.

    Creates multi-modal client distributions by sampling from a mixture
    of Dirichlet distributions with different concentration parameters.

    Args:
        num_clients: Number of clients.
        num_modes: Number of mixture components.
        alphas: List of Dirichlet alpha values for each mode.
        seed: Random seed.

    Example:
        >>> partitioner = MoDMPartitioner(
        ...     num_clients=20,
        ...     num_modes=3,
        ...     alphas=[0.1, 1.0, 10.0],
        ... )
        >>> client_indices = partitioner.partition(labels)
    """

    def __init__(
        self,
        num_clients: int,
        num_modes: int = 2,
        alphas: list[float] | None = None,
        seed: int = 42,
    ) -> None:
        """Initialize MoDM partitioner.

        Args:
            num_clients: Number of clients.
            num_modes: Number of mixture components.
            alphas: Alpha values per mode (default: linspace from 0.1 to 10).
            seed: Random seed.
        """
        super().__init__(num_clients, seed)
        self.num_modes = num_modes
        self.alphas = alphas or list(np.linspace(0.1, 10.0, num_modes))

    def partition(self, labels: np.ndarray | Any) -> dict[int, list[int]]:
        """Partition using mixture of Dirichlet distributions.

        Args:
            labels: Array of class labels.

        Returns:
            Dictionary mapping client IDs to sample indices.
        """
        labels = self._get_targets(labels)
        num_classes = len(np.unique(labels))
        rng = np.random.default_rng(self.seed)

        # Assign each client to a mode
        client_modes = rng.integers(0, self.num_modes, size=self.num_clients)

        # Sample Dirichlet proportions per client based on mode
        client_proportions = []
        for mode in client_modes:
            alpha = self.alphas[mode]
            props = rng.dirichlet(np.repeat(alpha, num_classes))
            client_proportions.append(props)

        # Get indices per class
        class_indices = [np.where(labels == c)[0] for c in range(num_classes)]
        for indices in class_indices:
            rng.shuffle(indices)

        client_indices: dict[int, list[int]] = {i: [] for i in range(self.num_clients)}

        # Distribute samples
        for c in range(num_classes):
            indices = class_indices[c]
            num_samples = len(indices)

            # Compute expected counts per client
            expected = np.array([p[c] for p in client_proportions])
            expected = expected / expected.sum()
            counts = (expected * num_samples).astype(int)

            # Handle remainder
            remainder = num_samples - counts.sum()
            for i in range(remainder):
                counts[i % self.num_clients] += 1

            # Assign indices
            curr = 0
            for client_id in range(self.num_clients):
                count = counts[client_id]
                assigned = indices[curr : curr + count].tolist()
                client_indices[client_id].extend(assigned)
                curr += count

        return client_indices
