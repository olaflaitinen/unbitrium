"""FedSim Aggregator implementation.

Similarity-guided aggregation that weights client contributions based
on cosine similarity to the global model.

Mathematical formulation:

$$
w^{t+1} = \\sum_k \\text{sim}(w_k^t, w_g^t) \\cdot w_k^t
$$

Author: Olaf Yunus Laitinen Imanov <oyli@dtu.dk>
License: EUPL-1.2
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from unbitrium.aggregators.base import Aggregator


class FedSim(Aggregator):
    """Similarity-guided federated aggregator.

    Weights client updates by their cosine similarity to the global model,
    reducing the influence of divergent clients.

    Args:
        similarity_threshold: Minimum similarity to include a client.
            Clients below this threshold are excluded.
        temperature: Temperature for softmax normalization of similarities.

    Example:
        >>> aggregator = FedSim(similarity_threshold=0.5)
        >>> new_model, metrics = aggregator.aggregate(updates, global_model)
    """

    def __init__(
        self,
        similarity_threshold: float = 0.0,
        temperature: float = 1.0,
    ) -> None:
        """Initialize FedSim aggregator.

        Args:
            similarity_threshold: Minimum cosine similarity threshold.
            temperature: Softmax temperature for weight normalization.
        """
        self.similarity_threshold = similarity_threshold
        self.temperature = temperature

    def _flatten_state_dict(self, state_dict: dict[str, torch.Tensor]) -> torch.Tensor:
        """Flatten model parameters into a single vector."""
        tensors = []
        for key, value in state_dict.items():
            if isinstance(value, torch.Tensor) and value.requires_grad is not False:
                tensors.append(value.view(-1).float())
        return torch.cat(tensors) if tensors else torch.tensor([])

    def _compute_similarity(
        self,
        client_state: dict[str, torch.Tensor],
        global_state: dict[str, torch.Tensor],
    ) -> float:
        """Compute cosine similarity between client and global model."""
        client_flat = self._flatten_state_dict(client_state)
        global_flat = self._flatten_state_dict(global_state)

        if client_flat.numel() == 0 or global_flat.numel() == 0:
            return 0.0

        similarity = F.cosine_similarity(
            client_flat.unsqueeze(0),
            global_flat.unsqueeze(0),
        )
        return similarity.item()

    def aggregate(
        self,
        updates: list[dict[str, Any]],
        current_global_model: torch.nn.Module,
    ) -> tuple[torch.nn.Module, dict[str, float]]:
        """Aggregate updates weighted by model similarity.

        Args:
            updates: List of client updates with 'state_dict' and 'num_samples'.
            current_global_model: Current global model.

        Returns:
            Tuple of (updated global model, aggregation metrics).
        """
        if not updates:
            return current_global_model, {"aggregated_clients": 0.0}

        global_state = current_global_model.state_dict()

        # Compute similarities
        similarities = []
        for update in updates:
            sim = self._compute_similarity(update["state_dict"], global_state)
            similarities.append(sim)

        # Filter by threshold
        valid_updates = []
        valid_sims = []
        for update, sim in zip(updates, similarities):
            if sim >= self.similarity_threshold:
                valid_updates.append(update)
                valid_sims.append(sim)

        if not valid_updates:
            return current_global_model, {"aggregated_clients": 0.0}

        # Normalize similarities as weights
        sim_tensor = torch.tensor(valid_sims) / self.temperature
        weights = F.softmax(sim_tensor, dim=0).tolist()

        # Aggregate
        new_state_dict: dict[str, torch.Tensor] = {}
        first_state = valid_updates[0]["state_dict"]

        for key in first_state.keys():
            if isinstance(first_state[key], torch.Tensor):
                weighted_sum = torch.zeros_like(first_state[key], dtype=torch.float32)
                for update, weight in zip(valid_updates, weights):
                    weighted_sum += update["state_dict"][key].float() * weight
                new_state_dict[key] = weighted_sum.to(first_state[key].dtype)
            else:
                new_state_dict[key] = first_state[key]

        current_global_model.load_state_dict(new_state_dict)

        metrics = {
            "num_participants": float(len(valid_updates)),
            "avg_similarity": float(sum(valid_sims) / len(valid_sims)),
            "excluded_clients": float(len(updates) - len(valid_updates)),
        }
        return current_global_model, metrics
