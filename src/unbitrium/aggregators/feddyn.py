"""FedDyn Aggregator implementation.

Dynamic regularization for improved convergence under heterogeneity.

Mathematical formulation:

$$
F_k(w) - \\langle a_k^t, w \\rangle + \\frac{\\alpha}{2} \\| w \\|^2
$$

Author: Olaf Yunus Laitinen Imanov <oyli@dtu.dk>
License: EUPL-1.2
"""

from __future__ import annotations

from typing import Any

import torch

from unbitrium.aggregators.base import Aggregator


class FedDyn(Aggregator):
    """FedDyn aggregator with dynamic regularization.

    Maintains per-client dual variables to correct for local drift,
    improving convergence in heterogeneous settings.

    Args:
        alpha: Regularization coefficient.
        num_clients: Total number of clients.

    Example:
        >>> aggregator = FedDyn(alpha=0.01, num_clients=100)
        >>> new_model, metrics = aggregator.aggregate(updates, global_model)
    """

    def __init__(self, alpha: float = 0.01, num_clients: int = 100) -> None:
        """Initialize FedDyn aggregator.

        Args:
            alpha: Regularization coefficient.
            num_clients: Total number of clients.
        """
        self.alpha = alpha
        self.num_clients = num_clients
        self._h: dict[str, torch.Tensor] | None = None  # Gradient correction term

    def aggregate(
        self,
        updates: list[dict[str, Any]],
        current_global_model: torch.nn.Module,
    ) -> tuple[torch.nn.Module, dict[str, float]]:
        """Aggregate with dynamic regularization correction.

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

        global_state = current_global_model.state_dict()

        # Initialize correction term if needed
        if self._h is None:
            self._h = {
                k: torch.zeros_like(v, dtype=torch.float32)
                for k, v in global_state.items()
                if isinstance(v, torch.Tensor)
            }

        # Compute weighted average
        new_state_dict: dict[str, torch.Tensor] = {}
        first_state = updates[0]["state_dict"]

        for key in first_state.keys():
            if isinstance(first_state[key], torch.Tensor):
                weighted_sum = torch.zeros_like(first_state[key], dtype=torch.float32)
                for update in updates:
                    weight = update["num_samples"] / total_samples
                    weighted_sum += update["state_dict"][key].float() * weight

                # Apply gradient correction
                if key in self._h:
                    correction = self._h[key] / self.alpha
                    weighted_sum = weighted_sum - correction

                new_state_dict[key] = weighted_sum.to(first_state[key].dtype)

                # Update correction term
                if key in self._h:
                    delta = global_state[key].float() - weighted_sum
                    self._h[key] = self._h[key] - self.alpha * delta
            else:
                new_state_dict[key] = first_state[key]

        current_global_model.load_state_dict(new_state_dict)

        metrics = {
            "num_participants": float(len(updates)),
            "total_samples": float(total_samples),
            "alpha": self.alpha,
        }
        return current_global_model, metrics
