"""
Quantity Skew Partitioner.
"""

from typing import Any, Dict, List
import numpy as np
from unbitrium.partitioning.base import Partitioner

class QuantitySkewPowerLaw(Partitioner):
    r"""
    Quantity Skew Partitioner using Power Law.

    $n_k \propto k^{-\gamma}$
    """

    def __init__(self, num_clients: int, gamma: float = 0.5, seed: int = 42):
        super().__init__(num_clients, seed)
        self.gamma = gamma

    def partition(self, dataset: Any) -> Dict[int, List[int]]:
        targets = self._get_targets(dataset)
        total_samples = len(targets)

        rng = np.random.default_rng(self.seed)
        all_indices = np.arange(total_samples)
        rng.shuffle(all_indices)

        # Calculate sizes
        ranks = np.arange(1, self.num_clients + 1)
        probs = ranks ** (-self.gamma)
        probs = probs / probs.sum()

        # Determine exact counts
        counts = np.floor(probs * total_samples).astype(int)
        # Fix rounding error
        diff = total_samples - counts.sum()
        counts[:diff] += 1

        client_indices = {}
        curr = 0
        for i in range(self.num_clients):
            c = counts[i]
            client_indices[i] = all_indices[curr : curr+c].tolist()
            curr += c

        return client_indices
