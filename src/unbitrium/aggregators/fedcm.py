"""FedCM Aggregator implementation.

Client-level momentum correction for stable federated training.

Mathematical formulation:

$$
v_k^{t+1} = \\beta v_k^t + \\nabla F_k(w_k^t)
$$

Author: Olaf Yunus Laitinen Imanov <oyli@dtu.dk>
License: EUPL-1.2
"""

from __future__ import annotations

from typing import Any

import torch

from unbitrium.aggregators.base import Aggregator


class FedCM(Aggregator):
    """FedCM aggregator with client-level momentum.

    Maintains momentum buffers per client to dampen update oscillations
    and improve training stability.

    Args:
        beta: Momentum coefficient (0 to 1).

    Example:
        >>> aggregator = FedCM(beta=0.9)
        >>> new_model, metrics = aggregator.aggregate(updates, global_model)
    """

    def __init__(self, beta: float = 0.9) -> None:
        """Initialize FedCM aggregator.

        Args:
            beta: Momentum coefficient.
        """
        self.beta = beta
        self._momentum: dict[int, dict[str, torch.Tensor]] = {}

    def aggregate(
        self,
        updates: list[dict[str, Any]],
        current_global_model: torch.nn.Module,
    ) -> tuple[torch.nn.Module, dict[str, float]]:
        """Aggregate with momentum-corrected updates.

        Args:
            updates: List of client updates with 'state_dict', 'num_samples',
                and optionally 'client_id'.
            current_global_model: Current global model.

        Returns:
            Tuple of (updated global model, aggregation metrics).
        """
        if not updates:
            return current_global_model, {"aggregated_clients": 0.0}

        total_samples = sum(u.get("num_samples", 0) for u in updates)
        if total_samples == 0:
            return current_global_model, {"aggregated_clients": 0.0}

        global_state = current_global_model.state_dict()

        # Apply momentum correction per client
        corrected_updates = []
        for i, update in enumerate(updates):
            client_id = update.get("client_id", i)

            if client_id not in self._momentum:
                # Initialize momentum buffer
                self._momentum[client_id] = {
                    k: torch.zeros_like(v, dtype=torch.float32)
                    for k, v in update["state_dict"].items()
                    if isinstance(v, torch.Tensor)
                }

            corrected_state: dict[str, torch.Tensor] = {}
            for key, value in update["state_dict"].items():
                if isinstance(value, torch.Tensor) and key in self._momentum[client_id]:
                    # Compute update delta
                    delta = value.float() - global_state[key].float()
                    # Apply momentum
                    self._momentum[client_id][key] = (
                        self.beta * self._momentum[client_id][key] + delta
                    )
                    corrected_state[key] = (
                        global_state[key].float() + self._momentum[client_id][key]
                    ).to(value.dtype)
                else:
                    corrected_state[key] = value

            corrected_updates.append(
                {
                    "state_dict": corrected_state,
                    "num_samples": update["num_samples"],
                }
            )

        # Aggregate corrected updates
        new_state_dict: dict[str, torch.Tensor] = {}
        first_state = corrected_updates[0]["state_dict"]

        for key in first_state.keys():
            if isinstance(first_state[key], torch.Tensor):
                weighted_sum = torch.zeros_like(first_state[key], dtype=torch.float32)
                for update in corrected_updates:
                    weight = update["num_samples"] / total_samples
                    weighted_sum += update["state_dict"][key].float() * weight
                new_state_dict[key] = weighted_sum.to(first_state[key].dtype)
            else:
                new_state_dict[key] = first_state[key]

        current_global_model.load_state_dict(new_state_dict)

        metrics = {
            "num_participants": float(len(updates)),
            "total_samples": float(total_samples),
            "beta": self.beta,
        }
        return current_global_model, metrics
