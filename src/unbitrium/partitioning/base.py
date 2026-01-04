"""Base partitioner interface for Unbitrium.

Defines the abstract base class for data partitioning strategies.

Author: Olaf Yunus Laitinen Imanov <oyli@dtu.dk>
License: EUPL-1.2
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class Partitioner(ABC):
    """Abstract base class for federated data partitioners.

    All partitioning strategies must inherit from this class and
    implement the partition method.

    Args:
        num_clients: Number of clients to partition data across.
        seed: Random seed for reproducibility.
    """

    def __init__(self, num_clients: int, seed: int = 42) -> None:
        """Initialize partitioner.

        Args:
            num_clients: Number of clients.
            seed: Random seed.
        """
        self.num_clients = num_clients
        self.seed = seed

    @abstractmethod
    def partition(self, dataset: Any) -> dict[int, list[int]]:
        """Partition dataset indices across clients.

        Args:
            dataset: Dataset or array of labels to partition.

        Returns:
            Dictionary mapping client IDs to lists of sample indices.
        """
        pass

    def _get_targets(self, dataset: Any) -> np.ndarray:
        """Extract targets/labels from dataset.

        Args:
            dataset: Dataset object with targets attribute or array.

        Returns:
            NumPy array of labels.
        """
        if hasattr(dataset, "targets"):
            targets = dataset.targets
        elif hasattr(dataset, "labels"):
            targets = dataset.labels
        elif hasattr(dataset, "y"):
            targets = dataset.y
        else:
            targets = dataset

        if hasattr(targets, "numpy"):
            return targets.numpy()
        return np.asarray(targets)
