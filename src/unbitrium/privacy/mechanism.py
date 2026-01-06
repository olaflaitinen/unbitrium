"""Privacy mechanisms for Unbitrium.

Implements differential privacy mechanisms for gradient perturbation.

Mathematical formulation:

Gaussian mechanism: $\\mathcal{M}(x) = f(x) + \\mathcal{N}(0, \\sigma^2)$

where $\\sigma = \\frac{\\Delta_2 \\sqrt{2 \\ln(1.25/\\delta)}}{\\epsilon}$

Author: Olaf Yunus Laitinen Imanov <oyli@dtu.dk>
License: EUPL-1.2
"""

from __future__ import annotations

import numpy as np
import torch


class GaussianMechanism:
    """Gaussian differential privacy mechanism.

    Adds calibrated Gaussian noise to achieve (epsilon, delta)-DP.

    Args:
        epsilon: Privacy budget.
        delta: Privacy parameter.
        sensitivity: L2 sensitivity of the function.

    Example:
        >>> mechanism = GaussianMechanism(epsilon=1.0, delta=1e-5, sensitivity=1.0)
        >>> noisy_grad = mechanism.apply(gradient)
    """

    def __init__(
        self,
        epsilon: float,
        delta: float = 1e-5,
        sensitivity: float = 1.0,
    ) -> None:
        """Initialize Gaussian mechanism.

        Args:
            epsilon: Privacy budget.
            delta: Privacy parameter.
            sensitivity: L2 sensitivity.
        """
        self.epsilon = epsilon
        self.delta = delta
        self.sensitivity = sensitivity
        self.sigma = self._compute_sigma()

    def _compute_sigma(self) -> float:
        """Compute noise scale for (epsilon, delta)-DP."""
        return self.sensitivity * np.sqrt(2 * np.log(1.25 / self.delta)) / self.epsilon

    def apply(self, tensor: torch.Tensor) -> torch.Tensor:
        """Apply Gaussian noise to tensor.

        Args:
            tensor: Input tensor.

        Returns:
            Noisy tensor.
        """
        noise = torch.randn_like(tensor) * self.sigma
        return tensor + noise

    def apply_to_state_dict(
        self,
        state_dict: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Apply noise to model state dictionary.

        Args:
            state_dict: Model state dictionary.

        Returns:
            Noised state dictionary.
        """
        return {
            k: self.apply(v) if isinstance(v, torch.Tensor) else v
            for k, v in state_dict.items()
        }


class LaplaceMechanism:
    """Laplace differential privacy mechanism.

    Adds calibrated Laplace noise to achieve epsilon-DP.

    Args:
        epsilon: Privacy budget.
        sensitivity: L1 sensitivity of the function.

    Example:
        >>> mechanism = LaplaceMechanism(epsilon=1.0, sensitivity=1.0)
        >>> noisy_value = mechanism.apply(value)
    """

    def __init__(self, epsilon: float, sensitivity: float = 1.0) -> None:
        """Initialize Laplace mechanism.

        Args:
            epsilon: Privacy budget.
            sensitivity: L1 sensitivity.
        """
        self.epsilon = epsilon
        self.sensitivity = sensitivity
        self.scale = sensitivity / epsilon

    def apply(self, tensor: torch.Tensor) -> torch.Tensor:
        """Apply Laplace noise to tensor.

        Args:
            tensor: Input tensor.

        Returns:
            Noisy tensor.
        """
        # Sample from Laplace distribution
        uniform = torch.rand_like(tensor) - 0.5
        noise = -self.scale * torch.sign(uniform) * torch.log1p(-2 * torch.abs(uniform))
        return tensor + noise


def clip_gradients(
    gradients: dict[str, torch.Tensor],
    max_norm: float,
) -> dict[str, torch.Tensor]:
    """Clip gradients to bound sensitivity.

    Args:
        gradients: Dictionary of gradient tensors.
        max_norm: Maximum L2 norm.

    Returns:
        Clipped gradients.
    """
    # Flatten and compute total norm
    tensors = [g.view(-1) for g in gradients.values() if isinstance(g, torch.Tensor)]
    if not tensors:
        return gradients

    total_norm = torch.norm(torch.cat(tensors))

    if total_norm > max_norm:
        scale = max_norm / total_norm
        return {
            k: v * scale if isinstance(v, torch.Tensor) else v
            for k, v in gradients.items()
        }
    return gradients
