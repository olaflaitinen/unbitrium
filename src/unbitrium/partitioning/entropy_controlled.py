"""
Entropy Controlled Partitioner.
"""

from typing import Any, Dict, List
import numpy as np
from unbitrium.partitioning.base import Partitioner
from unbitrium.partitioning.dirichlet import DirichletLabelSkew
from unbitrium.metrics.distribution import compute_label_entropy

class EntropyControlledPartition(Partitioner):
    """
    Ensures partitions meet a target entropy criteria (Hardness control).
    Implementation: Rejection Sampling or Optimization loop over Dirichlet alpha.
    """

    def __init__(self, num_clients: int, target_entropy: float, tolerance: float = 0.1, seed: int = 42):
        super().__init__(num_clients, seed)
        self.target_entropy = target_entropy
        self.tolerance = tolerance

    def partition(self, dataset: Any) -> Dict[int, List[int]]:
        """
        Attempts to find a partition where avg client entropy is close to target.
        Starts with alpha=0.1 (low entropy) and binary searches?

        Or just rejection sample partitioner seeds?
        Rejection sampling alpha is better.
        """

        low = 0.001
        high = 1000.0
        best_indices = None
        best_diff = float("inf")

        targets = self._get_targets(dataset)
        num_classes = len(np.unique(targets))
        max_entropy = np.log(num_classes) # nats

        if self.target_entropy > max_entropy:
            raise ValueError(f"Target entropy {self.target_entropy} > Max {max_entropy}")

        # Binary search for Alpha that yields the entropy
        for _ in range(10): # 10 iterations
            mid = (low + high) / 2
            p = DirichletLabelSkew(self.num_clients, alpha=mid, seed=self.seed)
            indices = p.partition(dataset)

            # Compute metric
            entropies = []
            for cids in indices.values():
                 # Histogram
                 tags = targets[cids]
                 counts = np.bincount(tags, minlength=num_classes)
                 dist = counts / counts.sum() if counts.sum() > 0 else counts
                 h = 0
                 for x in dist:
                     if x > 0:
                         h -= x * np.log(x)
                 entropies.append(h)

            avg_h = np.mean(entropies)
            diff = abs(avg_h - self.target_entropy)

            if diff < best_diff:
                best_diff = diff
                best_indices = indices

            if diff < self.tolerance:
                break

            # Dirichlet: High Alpha -> Uniform (Max Entropy)
            # Low Alpha -> Skewed (Low Entropy)
            if avg_h < self.target_entropy:
                # Too skewed, need more uniformity -> Increase Alpha
                low = mid
            else:
                high = mid

        return best_indices
