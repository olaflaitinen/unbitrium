"""
MoDM Partitioner.
"""

from typing import Any, Dict, List
import numpy as np
from unbitrium.partitioning.base import Partitioner

class MoDM(Partitioner):
    """
    Mixture-of-Dirichlet-Multinomials (MoDM).

    Models client distributions as a mixture.
    Allows creating groups of clients with distinct skew patterns.
    """

    def __init__(self, num_clients: int, num_mixtures: int = 2, alphas: List[float] = None, seed: int = 42):
        super().__init__(num_clients, seed)
        self.num_mixtures = num_mixtures
        self.alphas = alphas if alphas else [0.1] * num_mixtures

    def partition(self, dataset: Any) -> Dict[int, List[int]]:
        targets = self._get_targets(dataset)
        num_classes = len(np.unique(targets))

        client_indices = {i: [] for i in range(self.num_clients)}
        rng = np.random.default_rng(self.seed)

        # Assign clients to mixtures
        # Uniform assignment or parameter? Uniform for now.
        mixture_assignments = rng.integers(0, self.num_mixtures, size=self.num_clients)

        # Partition data class-wise per mixture?
        # A bit tricky. Easier: Sample target distribution for each CLIENT based on their mixture's alpha.
        # Then fill clients to satisfy that distribution.

        # 1. Generate Target Distributions (p_k)
        client_p = []
        for i in range(self.num_clients):
            m = mixture_assignments[i]
            alpha = self.alphas[m]
            p = rng.dirichlet(np.repeat(alpha, num_classes))
            client_p.append(p)

        # 2. Assign Samples
        # Naive approach: Iterate samples, assign to client proportional to p_k(y) and remaining capacity?
        # Or: Iterate clients, assign samples from pool.

        # Let's use sorting based assignment or "Simulated Sampling"
        # We need to respect global dataset limits.

        class_indices = [np.where(targets == c)[0] for c in range(num_classes)]
        for c_idx in class_indices:
            rng.shuffle(c_idx)

        cursors = [0] * num_classes

        # We assume dataset is large enough or we cycle.
        # To be strictly partitioning, we must split the available data.

        # Compute Desired Counts per client per class
        # Assuming equal data size per client for simplicity unless Quantity Skew combined (usually separate).
        n_per_client = len(targets) // self.num_clients

        for cid in range(self.num_clients):
            p = client_p[cid]
            desired_counts = rng.multinomial(n_per_client, p)

            for c, count in enumerate(desired_counts):
                # Take 'count' samples from class c
                start = cursors[c]
                # If we run out, wrap around or stop?
                # Strict partition = stop.
                # If we stop, last clients get nothing.
                # Better: Distribute available N_c according to sum of desired P across clients?

                # Simplified robust logic:
                # We allocated P vector. Now we allocate the *actual* available samples.
                pass

        # Re-implementation for Robust Partitioning:
        # Use matrix allocation.
        matrix = np.array(client_p) # (N_clients, N_classes)
        # Normalize columns so they sum to 1 (Prob that a sample of class y goes to client k)
        col_sums = matrix.sum(axis=0)
        prob_y_to_k = matrix / col_sums[None, :]

        for c in range(num_classes):
            indices = class_indices[c]
            # Split indices among clients based on column c of prob matrix
            probs = prob_y_to_k[:, c]
            counts = rng.multinomial(len(indices), probs)

            curr = 0
            for cid, count in enumerate(counts):
                assigned = indices[curr : curr+count]
                client_indices[cid].extend(assigned)
                curr += count

        return client_indices
