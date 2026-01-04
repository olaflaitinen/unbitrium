"""FedAdam Aggregator implementation.

Server-side Adam optimizer on aggregated updates.

Mathematical formulation:

$$
m_t = \\beta_1 m_{t-1} + (1 - \\beta_1) \\Delta_t
$$
$$
v_t = \\beta_2 v_{t-1} + (1 - \\beta_2) \\Delta_t^2
$$
$$
w_{t+1} = w_t - \\eta \\frac{m_t}{\\sqrt{v_t} + \\epsilon}
$$

Author: Olaf Yunus Laitinen Imanov <oyli@dtu.dk>
License: EUPL-1.2
"""

from __future__ import annotations

from typing import Any

import torch

from unbitrium.aggregators.base import Aggregator


class FedAdam(Aggregator):
    """FedAdam aggregator with server-side Adam optimization.

    Applies Adam optimizer to the aggregated pseudo-gradient on the server,
    improving convergence and reducing communication rounds.

    Args:
        lr: Server learning rate.
        beta1: First moment decay rate.
        beta2: Second moment decay rate.
        epsilon: Numerical stability constant.

    Example:
        >>> aggregator = FedAdam(lr=0.01, beta1=0.9, beta2=0.999)
        >>> new_model, metrics = aggregator.aggregate(updates, global_model)
    """

    def __init__(
        self,
        lr: float = 0.01,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
    ) -> None:
        """Initialize FedAdam aggregator.

        Args:
            lr: Server learning rate.
            beta1: First moment decay rate.
            beta2: Second moment decay rate.
            epsilon: Numerical stability constant.
        """
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self._m: dict[str, torch.Tensor] | None = None  # First moment
        self._v: dict[str, torch.Tensor] | None = None  # Second moment
        self._t = 0  # Timestep

    def aggregate(
        self,
        updates: list[dict[str, Any]],
        current_global_model: torch.nn.Module,
    ) -> tuple[torch.nn.Module, dict[str, float]]:
        """Aggregate with server-side Adam optimization.

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
        self._t += 1

        # Initialize moment buffers
        if self._m is None:
            self._m = {
                k: torch.zeros_like(v, dtype=torch.float32)
                for k, v in global_state.items()
                if isinstance(v, torch.Tensor)
            }
            self._v = {
                k: torch.zeros_like(v, dtype=torch.float32)
                for k, v in global_state.items()
                if isinstance(v, torch.Tensor)
            }

        # Compute weighted average (pseudo-gradient)
        avg_update: dict[str, torch.Tensor] = {}
        first_state = updates[0]["state_dict"]

        for key in first_state.keys():
            if isinstance(first_state[key], torch.Tensor):
                weighted_sum = torch.zeros_like(first_state[key], dtype=torch.float32)
                for update in updates:
                    weight = update["num_samples"] / total_samples
                    weighted_sum += update["state_dict"][key].float() * weight
                avg_update[key] = weighted_sum

        # Apply Adam update
        new_state_dict: dict[str, torch.Tensor] = {}
        for key in global_state.keys():
            if isinstance(global_state[key], torch.Tensor) and key in avg_update:
                # Pseudo-gradient (delta from global)
                delta = avg_update[key] - global_state[key].float()

                # Update moments
                self._m[key] = self.beta1 * self._m[key] + (1 - self.beta1) * delta
                self._v[key] = self.beta2 * self._v[key] + (1 - self.beta2) * (delta**2)

                # Bias correction
                m_hat = self._m[key] / (1 - self.beta1**self._t)
                v_hat = self._v[key] / (1 - self.beta2**self._t)

                # Update
                update_step = self.lr * m_hat / (torch.sqrt(v_hat) + self.epsilon)
                new_state_dict[key] = (
                    global_state[key].float() + update_step
                ).to(global_state[key].dtype)
            else:
                new_state_dict[key] = global_state[key]

        current_global_model.load_state_dict(new_state_dict)

        metrics = {
            "num_participants": float(len(updates)),
            "total_samples": float(total_samples),
            "timestep": float(self._t),
        }
        return current_global_model, metrics
