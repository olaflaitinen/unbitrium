"""AFL-DCS Aggregator implementation.

Asynchronous Federated Learning with Dynamic Client Scheduling.

Mathematical formulation:

$$
w^{t+1} = \\text{Agg}(\\{(w_k^t, s_k)\\}_k)
$$

where $s_k$ is the staleness of client $k$'s update.

Author: Olaf Yunus Laitinen Imanov <oyli@dtu.dk>
License: EUPL-1.2
"""

from __future__ import annotations

from typing import Any

import torch

from unbitrium.aggregators.base import Aggregator


class AFL_DCS(Aggregator):
    """Asynchronous FL with Dynamic Client Scheduling.

    Handles asynchronous client updates with staleness-aware weighting
    and dynamic scheduling to exclude stragglers.

    Args:
        max_staleness: Maximum allowed staleness before excluding update.
        staleness_decay: Decay factor for staleness weighting.

    Example:
        >>> aggregator = AFL_DCS(max_staleness=5, staleness_decay=0.9)
        >>> new_model, metrics = aggregator.aggregate(updates, global_model)
    """

    def __init__(
        self,
        max_staleness: int = 10,
        staleness_decay: float = 0.9,
    ) -> None:
        """Initialize AFL-DCS aggregator.

        Args:
            max_staleness: Maximum allowed staleness.
            staleness_decay: Exponential decay for staleness weighting.
        """
        self.max_staleness = max_staleness
        self.staleness_decay = staleness_decay
        self._global_round = 0

    def aggregate(
        self,
        updates: list[dict[str, Any]],
        current_global_model: torch.nn.Module,
    ) -> tuple[torch.nn.Module, dict[str, float]]:
        """Aggregate with staleness-aware weighting.

        Args:
            updates: List of client updates with 'state_dict', 'num_samples',
                and optionally 'round' (when update was computed).
            current_global_model: Current global model.

        Returns:
            Tuple of (updated global model, aggregation metrics).
        """
        if not updates:
            return current_global_model, {"aggregated_clients": 0.0}

        self._global_round += 1

        # Filter by staleness
        valid_updates = []
        for update in updates:
            update_round = update.get("round", self._global_round)
            staleness = self._global_round - update_round
            if staleness <= self.max_staleness:
                update["staleness"] = staleness
                valid_updates.append(update)

        if not valid_updates:
            return current_global_model, {"aggregated_clients": 0.0}

        # Compute staleness-adjusted weights
        weights = []
        for update in valid_updates:
            base_weight = update.get("num_samples", 1)
            staleness = update.get("staleness", 0)
            adjusted_weight = base_weight * (self.staleness_decay**staleness)
            weights.append(adjusted_weight)

        total_weight = sum(weights)
        if total_weight == 0:
            return current_global_model, {"aggregated_clients": 0.0}

        # Normalize weights
        weights = [w / total_weight for w in weights]

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

        avg_staleness = sum(u.get("staleness", 0) for u in valid_updates) / len(
            valid_updates
        )

        metrics = {
            "num_participants": float(len(valid_updates)),
            "excluded_stale": float(len(updates) - len(valid_updates)),
            "avg_staleness": avg_staleness,
            "global_round": float(self._global_round),
        }
        return current_global_model, metrics
