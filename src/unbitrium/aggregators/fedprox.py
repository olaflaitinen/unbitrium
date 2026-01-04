"""FedProx Aggregator implementation.

FedProx adds a proximal regularization term to local client objectives
to reduce client drift under heterogeneous data distributions.

Mathematical formulation:

$$
\\min_w F_k(w) + \\frac{\\mu}{2} \\| w - w_g \\|^2
$$

Author: Olaf Yunus Laitinen Imanov <oyli@dtu.dk>
License: EUPL-1.2
"""

from __future__ import annotations

from typing import Any

import torch

from unbitrium.aggregators.base import Aggregator


class FedProx(Aggregator):
    """FedProx aggregator with proximal regularization.

    The proximal term penalizes local model drift from the global model,
    improving convergence under non-IID data distributions.

    Args:
        mu: Proximal regularization coefficient. Higher values enforce
            stronger alignment with the global model.

    Example:
        >>> aggregator = FedProx(mu=0.01)
        >>> new_model, metrics = aggregator.aggregate(updates, global_model)
    """

    def __init__(self, mu: float = 0.01) -> None:
        """Initialize FedProx aggregator.

        Args:
            mu: Proximal regularization coefficient.
        """
        self.mu = mu

    def aggregate(
        self,
        updates: list[dict[str, Any]],
        current_global_model: torch.nn.Module,
    ) -> tuple[torch.nn.Module, dict[str, float]]:
        """Aggregate updates using weighted average (same as FedAvg).

        Note: The proximal term is applied during local training, not
        during aggregation. Aggregation is identical to FedAvg.

        Args:
            updates: List of client updates with 'state_dict' and 'num_samples'.
            current_global_model: Current global model.

        Returns:
            Tuple of (updated global model, aggregation metrics).
        """
        if not updates:
            return current_global_model, {"aggregated_clients": 0.0}

        total_samples = sum(u.get("num_samples", 0) for u in updates)
        if total_samples == 0:
            return current_global_model, {"aggregated_clients": 0.0}

        new_state_dict: dict[str, torch.Tensor] = {}
        first_state = updates[0]["state_dict"]

        for key in first_state.keys():
            if isinstance(first_state[key], torch.Tensor):
                weighted_sum = torch.zeros_like(first_state[key], dtype=torch.float32)
                for update in updates:
                    weight = update["num_samples"] / total_samples
                    weighted_sum += update["state_dict"][key].float() * weight
                new_state_dict[key] = weighted_sum.to(first_state[key].dtype)
            else:
                new_state_dict[key] = first_state[key]

        current_global_model.load_state_dict(new_state_dict)

        metrics = {
            "num_participants": float(len(updates)),
            "total_samples": float(total_samples),
            "mu": self.mu,
        }
        return current_global_model, metrics
