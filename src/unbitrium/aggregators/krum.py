"""
Krum Aggregator.
"""

from typing import Any, Dict, List, Tuple
import torch
from unbitrium.aggregators.base import Aggregator

class Krum(Aggregator):
    """
    Krum Aggregator.

    Selects one update $u_{k*}$ that minimizes the sum of squared distances
    to its $n - f - 2$ nearest neighbors.

    Robust against $f$ Byzantine attackers.
    """

    def __init__(self, num_byzantine: int = 0):
        self.f = num_byzantine

    def aggregate(
        self,
        updates: List[Dict[str, Any]],
        current_global_model: torch.nn.Module
    ) -> Tuple[torch.nn.Module, Dict[str, float]]:

        if not updates:
            return current_global_model, {}

        n = len(updates)
        f = self.f

        # Valid range for Krum: n >= 2f + 3
        # If not met, we can still run but guarantees fail.
        # m = n - f - 2 (Number of neighbors to sum distances to)
        m = n - f - 2
        if m < 1:
             m = n - 1 # Fallback, compare to all others (basically mean-like selection?)

        # Flatten updates
        flat_updates = [] # List of Tensors
        for u in updates:
            flat = self._flatten(u["state_dict"])
            flat_updates.append(flat)

        distances = torch.zeros((n, n))

        # Compute pairwise distances
        # O(n^2 d). Expensive.
        for i in range(n):
            for j in range(i + 1, n):
                d = torch.norm(flat_updates[i] - flat_updates[j]) ** 2
                distances[i, j] = d
                distances[j, i] = d

        scores = []
        for i in range(n):
            # Sort distances from i
            dists = distances[i]
            sorted_dists, _ = torch.sort(dists)
            # Sum smallest m (excluding self which is 0)
            # sorted_dists[0] is self (0.0)
            # Neighbours are indices 1..m
            score = torch.sum(sorted_dists[1 : m + 1])
            scores.append(score)

        # Select winner
        best_idx = torch.argmin(torch.tensor(scores)).item()

        # We can implement Multi-Krum (average top k), but spec says Krum (single)

        best_update = updates[best_idx]
        current_global_model.load_state_dict(best_update["state_dict"])

        return current_global_model, {"selected_client_idx": best_idx, "krum_score": scores[best_idx].item()}

    def _flatten(self, state_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        keys = sorted(state_dict.keys())
        tensors = [state_dict[k].float().view(-1) for k in keys]
        return torch.cat(tensors)
