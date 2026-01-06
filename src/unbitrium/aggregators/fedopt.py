"""FedOpt family of server-side optimization aggregators.

Provides FedAdam, FedYogi, and FedAdagrad for adaptive server-side updates.

Author: Olaf Yunus Laitinen Imanov <oyli@dtu.dk>
License: EUPL-1.2
"""

from __future__ import annotations

from typing import Any

import torch

from unbitrium.aggregators.base import Aggregator


class FedOpt(Aggregator):
    """Base class for FedOpt server-side optimization aggregators.

    Args:
        learning_rate: Server learning rate.
        beta1: First moment decay rate.
        beta2: Second moment decay rate.
        tau: Stability constant.
    """

    def __init__(
        self,
        learning_rate: float = 0.01,
        beta1: float = 0.9,
        beta2: float = 0.999,
        tau: float = 1e-3,
    ) -> None:
        """Initialize FedOpt aggregator."""
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.tau = tau
        self._m: dict[str, torch.Tensor] | None = None
        self._v: dict[str, torch.Tensor] | None = None
        self._t = 0

    def aggregate(
        self,
        updates: list[dict[str, Any]],
        current_global_model: torch.nn.Module,
    ) -> tuple[torch.nn.Module, dict[str, float]]:
        """Aggregate with server-side optimization.

        Args:
            updates: Client updates.
            current_global_model: Current global model.

        Returns:
            Updated model and metrics.
        """
        if not updates:
            return current_global_model, {"aggregated_clients": 0.0}

        total_samples = sum(u.get("num_samples", 1) for u in updates)
        global_state = current_global_model.state_dict()
        self._t += 1

        # Initialize momentum buffers
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

        # Compute weighted average
        avg_update: dict[str, torch.Tensor] = {}
        first_state = updates[0]["state_dict"]

        for key in first_state.keys():
            if isinstance(first_state[key], torch.Tensor):
                weighted_sum = torch.zeros_like(first_state[key], dtype=torch.float32)
                for update in updates:
                    weight = update.get("num_samples", 1) / total_samples
                    weighted_sum += update["state_dict"][key].float() * weight
                avg_update[key] = weighted_sum

        # Subclass-specific update
        new_state = self._apply_update(global_state, avg_update)
        current_global_model.load_state_dict(new_state)

        return current_global_model, {
            "num_participants": float(len(updates)),
            "timestep": float(self._t),
        }

    def _apply_update(
        self,
        global_state: dict[str, torch.Tensor],
        avg_update: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Apply optimization step. Override in subclasses."""
        return {k: avg_update.get(k, v) for k, v in global_state.items()}


class FedYogi(FedOpt):
    """FedYogi adaptive server-side optimizer.

    Uses sign-based second moment update for improved convergence.
    """

    def _apply_update(
        self,
        global_state: dict[str, torch.Tensor],
        avg_update: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Apply FedYogi update."""
        new_state: dict[str, torch.Tensor] = {}

        for key in global_state.keys():
            if isinstance(global_state[key], torch.Tensor) and key in avg_update:
                delta = avg_update[key] - global_state[key].float()

                # Ensure momentum buffers are initialized
                assert self._m is not None and self._v is not None

                # Update first moment
                self._m[key] = self.beta1 * self._m[key] + (1 - self.beta1) * delta

                # Yogi second moment update
                delta_sq = delta**2
                sign = torch.sign(delta_sq - self._v[key])
                self._v[key] = self._v[key] + (1 - self.beta2) * sign * delta_sq

                # Update
                update = self.lr * self._m[key] / (torch.sqrt(self._v[key]) + self.tau)
                new_state[key] = (global_state[key].float() + update).to(
                    global_state[key].dtype
                )
            else:
                new_state[key] = global_state[key]

        return new_state


class FedAdagrad(FedOpt):
    """FedAdagrad adaptive server-side optimizer.

    Accumulates squared gradients for adaptive learning rates.
    """

    def _apply_update(
        self,
        global_state: dict[str, torch.Tensor],
        avg_update: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Apply FedAdagrad update."""
        new_state: dict[str, torch.Tensor] = {}

        for key in global_state.keys():
            if isinstance(global_state[key], torch.Tensor) and key in avg_update:
                delta = avg_update[key] - global_state[key].float()

                # Ensure momentum buffers are initialized
                assert self._v is not None

                # Accumulate squared gradient
                self._v[key] = self._v[key] + delta**2

                # Update
                update = self.lr * delta / (torch.sqrt(self._v[key]) + self.tau)
                new_state[key] = (global_state[key].float() + update).to(
                    global_state[key].dtype
                )
            else:
                new_state[key] = global_state[key]

        return new_state
