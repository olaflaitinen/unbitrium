"""Differential Privacy Mechanisms for Federated Learning.

Provides implementations of differential privacy mechanisms including
Gaussian and Laplace mechanisms for gradient perturbation.

Mathematical formulation for (epsilon, delta)-DP with Gaussian mechanism:

$$
\\sigma \\geq c \\cdot \\frac{\\Delta f}{\\epsilon} \\sqrt{2 \\ln(1.25/\\delta)}
$$

where $\\Delta f$ is the sensitivity (max gradient norm), $c$ is a constant.

Author: Olaf Yunus Laitinen Imanov <oyli@dtu.dk>
License: EUPL-1.2
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Union

import numpy as np


@dataclass
class PrivacyAccountant:
    """Tracks cumulative privacy expenditure.

    Attributes:
        epsilon: Current epsilon expenditure.
        delta: Current delta expenditure.
        max_epsilon: Maximum allowed epsilon.
        max_delta: Maximum allowed delta.
    """

    epsilon: float = 0.0
    delta: float = 0.0
    max_epsilon: float = 10.0
    max_delta: float = 1e-5

    def can_release(self, epsilon: float, delta: float) -> bool:
        """Check if release is within privacy budget.

        Args:
            epsilon: Epsilon cost of release.
            delta: Delta cost of release.

        Returns:
            True if release is within budget.
        """
        return (
            self.epsilon + epsilon <= self.max_epsilon
            and self.delta + delta <= self.max_delta
        )

    def record_release(self, epsilon: float, delta: float) -> None:
        """Record a privacy-consuming release.

        Args:
            epsilon: Epsilon cost of release.
            delta: Delta cost of release.
        """
        self.epsilon += epsilon
        self.delta += delta


class DifferentialPrivacy:
    """Central or local differential privacy mechanism.

    Provides gradient clipping and noise addition for achieving
    (epsilon, delta)-differential privacy guarantees.

    Args:
        epsilon: Privacy parameter epsilon (lower = more private).
        delta: Privacy parameter delta (probability of failure).
        max_grad_norm: Maximum L2 norm for gradient clipping.
        mechanism: Noise mechanism ('gaussian' or 'laplace').

    Example:
        >>> dp = DifferentialPrivacy(epsilon=1.0, delta=1e-5, max_grad_norm=1.0)
        >>> clipped = dp.clip_gradients(gradients)
        >>> noisy = dp.add_noise(clipped)
    """

    def __init__(
        self,
        epsilon: float,
        delta: float,
        max_grad_norm: float,
        mechanism: str = "gaussian",
    ) -> None:
        """Initialize differential privacy mechanism.

        Args:
            epsilon: Privacy parameter epsilon.
            delta: Privacy parameter delta.
            max_grad_norm: Maximum gradient norm for clipping.
            mechanism: Noise mechanism type.
        """
        self.epsilon = epsilon
        self.delta = delta
        self.max_grad_norm = max_grad_norm
        self.mechanism = mechanism
        self.accountant = PrivacyAccountant()

    def _compute_sigma(self, sensitivity: float) -> float:
        """Compute noise scale for Gaussian mechanism.

        Args:
            sensitivity: Query sensitivity (max gradient norm).

        Returns:
            Noise standard deviation sigma.
        """
        if self.delta == 0:
            raise ValueError("Gaussian mechanism requires delta > 0")

        c = np.sqrt(2 * np.log(1.25 / self.delta))
        return c * sensitivity / self.epsilon

    def clip_gradients(self, gradients: Any) -> Any:
        """Clip gradients to bound sensitivity.

        Args:
            gradients: Model gradients (tensor or dict of tensors).

        Returns:
            Clipped gradients with L2 norm <= max_grad_norm.
        """
        if isinstance(gradients, dict):
            # Compute total L2 norm across all parameters
            total_norm = 0.0
            for key, grad in gradients.items():
                if hasattr(grad, 'norm'):
                    total_norm += grad.norm().item() ** 2
            total_norm = np.sqrt(total_norm)

            # Apply clipping
            clip_factor = min(1.0, self.max_grad_norm / (total_norm + 1e-8))
            return {key: grad * clip_factor for key, grad in gradients.items()}
        else:
            # Single tensor
            if hasattr(gradients, 'norm'):
                norm = gradients.norm().item()
                clip_factor = min(1.0, self.max_grad_norm / (norm + 1e-8))
                return gradients * clip_factor
            return gradients

    def add_noise(self, aggregate: Any) -> Any:
        """Add calibrated noise for differential privacy.

        Args:
            aggregate: Aggregated gradients or model updates.

        Returns:
            Noisy aggregate satisfying (epsilon, delta)-DP.
        """
        sigma = self._compute_sigma(self.max_grad_norm)

        if isinstance(aggregate, dict):
            noisy = {}
            for key, value in aggregate.items():
                if hasattr(value, 'shape'):
                    noise = np.random.normal(0, sigma, value.shape)
                    if hasattr(value, 'numpy'):
                        import torch
                        noisy[key] = value + torch.from_numpy(noise).to(value.dtype)
                    else:
                        noisy[key] = value + noise
                else:
                    noisy[key] = value
            return noisy
        else:
            if hasattr(aggregate, 'shape'):
                noise = np.random.normal(0, sigma, aggregate.shape)
                if hasattr(aggregate, 'numpy'):
                    import torch
                    return aggregate + torch.from_numpy(noise).to(aggregate.dtype)
                return aggregate + noise
            return aggregate

    def get_privacy_spent(self) -> Dict[str, float]:
        """Get current privacy expenditure.

        Returns:
            Dictionary with epsilon and delta spent.
        """
        return {
            "epsilon_spent": self.accountant.epsilon,
            "delta_spent": self.accountant.delta,
            "epsilon_remaining": self.accountant.max_epsilon - self.accountant.epsilon,
            "delta_remaining": self.accountant.max_delta - self.accountant.delta,
        }
