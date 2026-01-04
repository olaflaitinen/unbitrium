"""FedAvg Aggregator implementation.

Federated Averaging (FedAvg) is the foundational aggregation algorithm
for federated learning, computing a weighted average of client model updates.

Mathematical formulation:

$$
w^{t+1} = \\sum_{k=1}^K \\frac{n_k}{N} w_k^t
$$

where $n_k$ is the number of samples on client $k$, and $N = \\sum_k n_k$.

Author: Olaf Yunus Laitinen Imanov <oyli@dtu.dk>
License: EUPL-1.2
"""

from __future__ import annotations

from typing import Any

import torch

from unbitrium.aggregators.base import Aggregator


class FedAvg(Aggregator):
    """Federated Averaging aggregator.

    Computes a weighted average of client model updates where weights
    are proportional to local dataset sizes.

    Complexity:
        Time: O(K * P) where K is number of clients, P is number of parameters
        Space: O(P) for the aggregated model

    Example:
        >>> aggregator = FedAvg()
        >>> updates = [
        ...     {"state_dict": model1.state_dict(), "num_samples": 100},
        ...     {"state_dict": model2.state_dict(), "num_samples": 200},
        ... ]
        >>> new_model, metrics = aggregator.aggregate(updates, global_model)
    """

    def aggregate(
        self,
        updates: list[dict[str, Any]],
        current_global_model: torch.nn.Module,
    ) -> tuple[torch.nn.Module, dict[str, float]]:
        """Aggregate model updates using weighted average.

        Args:
            updates: List of dictionaries containing 'state_dict' and 'num_samples'.
            current_global_model: The global model at the current round.

        Returns:
            Tuple of (updated global model, aggregation metrics).

        Raises:
            ValueError: If updates is empty or contains invalid data.
        """
        if not updates:
            return current_global_model, {"aggregated_clients": 0.0}

        total_samples = sum(u.get("num_samples", 0) for u in updates)
        if total_samples == 0:
            return current_global_model, {"aggregated_clients": 0.0}

        # Initialize aggregated state dict
        new_state_dict: dict[str, torch.Tensor] = {}
        first_state = updates[0]["state_dict"]

        for key in first_state.keys():
            if isinstance(first_state[key], torch.Tensor):
                # Compute weighted sum
                weighted_sum = torch.zeros_like(first_state[key], dtype=torch.float32)
                for update in updates:
                    weight = update["num_samples"] / total_samples
                    weighted_sum += update["state_dict"][key].float() * weight
                new_state_dict[key] = weighted_sum.to(first_state[key].dtype)
            else:
                # Non-tensor values (buffers, metadata)
                new_state_dict[key] = first_state[key]

        current_global_model.load_state_dict(new_state_dict)

        metrics = {
            "num_participants": float(len(updates)),
            "total_samples": float(total_samples),
        }
        return current_global_model, metrics
