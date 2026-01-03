"""
Dirichlet Label Skew Partitioner.
"""

from typing import Dict, List, Optional
import numpy as np
from unbitrium.partitioning.base import Partitioner

class DirichletLabelSkew(Partitioner):
    """
    Dirichlet partitioning ensuring label skew.

    Each client $k$ has label distribution $p_k \sim Dir(\alpha)$.
    """

    def __init__(self, num_clients: int, alpha: float = 0.5, seed: int = 42):
        super().__init__(num_clients, seed)
        self.alpha = alpha

    def partition(self, dataset: Any) -> Dict[int, List[int]]:
        """
        Partitions the dataset.
        Assumption: sub-class/base helper extracts targets.
        """
        targets = self._get_targets(dataset)
        num_classes = len(np.unique(targets))
        num_samples = len(targets)

        # Get indices per class
        class_indices = [np.where(targets == c)[0] for c in range(num_classes)]

        client_indices = {i: [] for i in range(self.num_clients)}

        rng = np.random.default_rng(self.seed)

        # Distribute class by class
        for c in range(num_classes):
            k = len(class_indices[c])
            # Sample proportions for this class across clients from Dir(alpha)
            proportions = rng.dirichlet(np.repeat(self.alpha, self.num_clients))

            # Convert props to counts (noisy rounding)
            # Use multinomial to convert float props to integers summing to k
            proportions = np.array([p * (len(class_indices[c]) < num_samples / num_classes) for p in proportions])
            # Re-normalize to avoid zero sum issues?
            proportions = proportions / proportions.sum()

            counts = rng.multinomial(k, proportions)

            # Shuffle indices of this class
            indices = class_indices[c].copy()
            rng.shuffle(indices)

            # Assign
            curr = 0
            for client_id in range(self.num_clients):
                count = counts[client_id]
                assigned = indices[curr : curr+count]
                client_indices[client_id].extend(assigned)
                curr += count

        return client_indices
