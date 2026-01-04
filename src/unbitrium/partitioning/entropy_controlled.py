"""Entropy Controlled Partitioner implementation.

Partitions data to achieve target label entropy per client.

Author: Olaf Yunus Laitinen Imanov <oyli@dtu.dk>
License: EUPL-1.2
"""

from __future__ import annotations

from typing import Any

import numpy as np

from unbitrium.partitioning.base import Partitioner


class EntropyControlledPartitioner(Partitioner):
    """Entropy-controlled partitioner.

    Creates client distributions targeting a specific label entropy,
    allowing precise control over heterogeneity levels.

    Args:
        num_clients: Number of clients.
        target_entropy: Target Shannon entropy (higher = more balanced).
        seed: Random seed.

    Example:
        >>> partitioner = EntropyControlledPartitioner(
        ...     num_clients=10,
        ...     target_entropy=1.5,
        ... )
        >>> client_indices = partitioner.partition(labels)
    """

    def __init__(
        self,
        num_clients: int,
        target_entropy: float = 1.5,
        seed: int = 42,
    ) -> None:
        """Initialize entropy controlled partitioner.

        Args:
            num_clients: Number of clients.
            target_entropy: Target label entropy.
            seed: Random seed.
        """
        super().__init__(num_clients, seed)
        self.target_entropy = target_entropy

    def partition(self, labels: np.ndarray | Any) -> dict[int, list[int]]:
        """Partition to achieve target entropy.

        Args:
            labels: Array of class labels.

        Returns:
            Dictionary mapping client IDs to sample indices.
        """
        labels = self._get_targets(labels)
        num_classes = len(np.unique(labels))
        rng = np.random.default_rng(self.seed)

        # Compute max entropy for reference
        max_entropy = np.log(num_classes)

        # Estimate Dirichlet alpha from target entropy
        # Higher alpha -> higher entropy (more uniform)
        # Lower alpha -> lower entropy (more concentrated)
        normalized_entropy = self.target_entropy / max_entropy
        alpha = self._entropy_to_alpha(normalized_entropy, num_classes)

        # Sample proportions with estimated alpha
        class_indices = [np.where(labels == c)[0] for c in range(num_classes)]
        for indices in class_indices:
            rng.shuffle(indices)

        client_indices: dict[int, list[int]] = {i: [] for i in range(self.num_clients)}

        for c in range(num_classes):
            indices = class_indices[c]
            num_samples = len(indices)

            if num_samples == 0:
                continue

            # Sample Dirichlet proportions
            proportions = rng.dirichlet(np.repeat(alpha, self.num_clients))
            counts = (proportions * num_samples).astype(int)

            # Handle remainder
            remainder = num_samples - counts.sum()
            for i in range(remainder):
                counts[i % self.num_clients] += 1

            # Assign
            curr = 0
            for client_id in range(self.num_clients):
                count = counts[client_id]
                assigned = indices[curr : curr + count].tolist()
                client_indices[client_id].extend(assigned)
                curr += count

        return client_indices

    def _entropy_to_alpha(self, norm_entropy: float, num_classes: int) -> float:
        """Estimate Dirichlet alpha from normalized entropy.

        Args:
            norm_entropy: Target entropy normalized by max entropy.
            num_classes: Number of classes.

        Returns:
            Estimated alpha value.
        """
        # Empirical mapping (approximation)
        # norm_entropy close to 1 -> alpha high (uniform)
        # norm_entropy close to 0 -> alpha low (concentrated)
        if norm_entropy >= 0.99:
            return 100.0
        if norm_entropy <= 0.01:
            return 0.01
        return 10.0 ** (4 * norm_entropy - 2)
