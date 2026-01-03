"""
Base class for data partitioners.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import numpy as np

class Partitioner(ABC):
    """
    Abstract base class for data partitioners.
    """

    def __init__(self, num_clients: int, seed: int = 42):
        self.num_clients = num_clients
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    @abstractmethod
    def partition(self, dataset: Any) -> Dict[int, List[int]]:
        """
        Partitions the dataset into client subsets.

        Parameters
        ----------
        dataset : Any
            The dataset to partition (must expose targets/labels).

        Returns
        -------
        Dict[int, List[int]]
            Map from client_id to list of data indices.
        """
        pass

    def _get_targets(self, dataset: Any) -> np.ndarray:
        """Helper to extract targets from standard datasets."""
        if hasattr(dataset, "targets"):
            return np.array(dataset.targets)
        elif hasattr(dataset, "labels"):
            return np.array(dataset.labels)
        # Fallback for subsets or custom dataset classes
        # ... logic ...
        return np.array([])
