"""pFedSim Aggregator implementation.

Personalized similarity-guided aggregation that combines global model
aggregation with client-specific personalization layers.

Mathematical formulation:

$$
\\theta_g^{t+1} = \\sum_k \\omega_k(\\text{sim}) \\cdot \\theta_{k,\\text{shared}}^t
$$

Author: Olaf Yunus Laitinen Imanov <oyli@dtu.dk>
License: EUPL-1.2
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from unbitrium.aggregators.base import Aggregator


class PFedSim(Aggregator):
    """Personalized similarity-guided federated aggregator.

    Extends FedSim with personalization by decoupling shared and
    client-specific model layers during aggregation.

    Args:
        similarity_threshold: Minimum similarity to include a client.
        personalization_weight: Weight given to personalized layers.
        shared_layer_prefix: Prefix for shared layer names (default: all layers).

    Example:
        >>> aggregator = PFedSim(similarity_threshold=0.5, personalization_weight=0.3)
        >>> new_model, metrics = aggregator.aggregate(updates, global_model)
    """

    def __init__(
        self,
        similarity_threshold: float = 0.0,
        personalization_weight: float = 0.3,
        shared_layer_prefix: str | None = None,
    ) -> None:
        """Initialize pFedSim aggregator.

        Args:
            similarity_threshold: Minimum cosine similarity threshold.
            personalization_weight: Weight for personalized layers.
            shared_layer_prefix: Prefix identifying shared layers.
        """
        self.similarity_threshold = similarity_threshold
        self.personalization_weight = personalization_weight
        self.shared_layer_prefix = shared_layer_prefix

    def _flatten_state_dict(self, state_dict: dict[str, torch.Tensor]) -> torch.Tensor:
        """Flatten model parameters into a single vector."""
        tensors = []
        for key, value in state_dict.items():
            if isinstance(value, torch.Tensor):
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

    def _is_shared_layer(self, key: str) -> bool:
        """Check if a layer key belongs to shared layers."""
        if self.shared_layer_prefix is None:
            return True
        return key.startswith(self.shared_layer_prefix)

    def aggregate(
        self,
        updates: list[dict[str, Any]],
        current_global_model: torch.nn.Module,
    ) -> tuple[torch.nn.Module, dict[str, float]]:
        """Aggregate shared layers with similarity weighting.

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
        weights = F.softmax(torch.tensor(valid_sims), dim=0).tolist()

        # Aggregate shared layers
        new_state_dict: dict[str, torch.Tensor] = {}
        first_state = valid_updates[0]["state_dict"]

        for key in first_state.keys():
            if isinstance(first_state[key], torch.Tensor):
                if self._is_shared_layer(key):
                    # Aggregate shared layers
                    weighted_sum = torch.zeros_like(
                        first_state[key], dtype=torch.float32
                    )
                    for update, weight in zip(valid_updates, weights):
                        weighted_sum += update["state_dict"][key].float() * weight
                    new_state_dict[key] = weighted_sum.to(first_state[key].dtype)
                else:
                    # Keep global model for non-shared layers
                    new_state_dict[key] = global_state[key]
            else:
                new_state_dict[key] = first_state[key]

        current_global_model.load_state_dict(new_state_dict)

        metrics = {
            "num_participants": float(len(valid_updates)),
            "avg_similarity": float(sum(valid_sims) / len(valid_sims)),
            "personalization_weight": self.personalization_weight,
        }
        return current_global_model, metrics
