"""Krum Aggregator implementation.

Robust aggregation selecting the update with minimal distance to others.

Mathematical formulation:

$$
k^* = \\arg\\min_k \\sum_{i \\in \\mathcal{N}_k} \\| u_k - u_i \\|^2
$$

where $\\mathcal{N}_k$ is the set of $K - f - 2$ nearest neighbors.

Author: Olaf Yunus Laitinen Imanov <oyli@dtu.dk>
License: EUPL-1.2
"""

from __future__ import annotations

from typing import Any

import torch

from unbitrium.aggregators.base import Aggregator


class Krum(Aggregator):
    """Krum robust aggregator.

    Selects the client update with minimum score, where score is the
    sum of distances to nearest neighbors. Robust against Byzantine failures.

    Args:
        num_byzantine: Expected number of Byzantine (malicious) clients.
        multi_krum: If > 1, average the top multi_krum updates.

    Example:
        >>> aggregator = Krum(num_byzantine=2)
        >>> new_model, metrics = aggregator.aggregate(updates, global_model)
    """

    def __init__(self, num_byzantine: int = 0, multi_krum: int = 1) -> None:
        """Initialize Krum aggregator.

        Args:
            num_byzantine: Expected number of Byzantine clients.
            multi_krum: Number of updates to select and average.
        """
        self.num_byzantine = num_byzantine
        self.multi_krum = multi_krum

    def _flatten_state_dict(self, state_dict: dict[str, torch.Tensor]) -> torch.Tensor:
        """Flatten model parameters into a single vector."""
        tensors = []
        for value in state_dict.values():
            if isinstance(value, torch.Tensor):
                tensors.append(value.view(-1).float())
        return torch.cat(tensors) if tensors else torch.tensor([])

    def aggregate(
        self,
        updates: list[dict[str, Any]],
        current_global_model: torch.nn.Module,
    ) -> tuple[torch.nn.Module, dict[str, float]]:
        """Aggregate using Krum selection.

        Args:
            updates: List of client updates with 'state_dict' and 'num_samples'.
            current_global_model: Current global model.

        Returns:
            Tuple of (updated global model, aggregation metrics).
        """
        if not updates:
            return current_global_model, {"aggregated_clients": 0.0}

        num_clients = len(updates)
        if num_clients <= 2 * self.num_byzantine + 2:
            # Fall back to averaging if not enough clients
            return self._simple_average(updates, current_global_model)

        # Flatten all updates
        flat_updates = [self._flatten_state_dict(u["state_dict"]) for u in updates]

        # Compute pairwise distances
        distances = torch.zeros(num_clients, num_clients)
        for i in range(num_clients):
            for j in range(i + 1, num_clients):
                dist = torch.norm(flat_updates[i] - flat_updates[j]).item()
                distances[i, j] = dist
                distances[j, i] = dist

        # Compute Krum scores
        num_neighbors = num_clients - self.num_byzantine - 2
        scores = []
        for i in range(num_clients):
            dists = distances[i].clone()
            dists[i] = float("inf")  # Exclude self
            nearest, _ = torch.topk(dists, num_neighbors, largest=False)
            scores.append(nearest.sum().item())

        # Select top updates
        _, indices_tensor = torch.topk(
            torch.tensor(scores), self.multi_krum, largest=False
        )
        selected_indices: list[int] = indices_tensor.tolist()

        # Average selected updates
        selected_updates = [updates[i] for i in selected_indices]
        return self._simple_average(selected_updates, current_global_model)

    def _simple_average(
        self,
        updates: list[dict[str, Any]],
        current_global_model: torch.nn.Module,
    ) -> tuple[torch.nn.Module, dict[str, float]]:
        """Simple average fallback."""
        if not updates:
            return current_global_model, {"aggregated_clients": 0.0}

        total_samples = sum(u.get("num_samples", 1) for u in updates)

        new_state_dict: dict[str, torch.Tensor] = {}
        first_state = updates[0]["state_dict"]

        for key in first_state.keys():
            if isinstance(first_state[key], torch.Tensor):
                weighted_sum = torch.zeros_like(first_state[key], dtype=torch.float32)
                for update in updates:
                    weight = update.get("num_samples", 1) / total_samples
                    weighted_sum += update["state_dict"][key].float() * weight
                new_state_dict[key] = weighted_sum.to(first_state[key].dtype)
            else:
                new_state_dict[key] = first_state[key]

        current_global_model.load_state_dict(new_state_dict)

        return current_global_model, {"num_participants": float(len(updates))}
