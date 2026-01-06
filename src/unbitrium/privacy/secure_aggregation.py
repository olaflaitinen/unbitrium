"""Secure Aggregation Interface for Federated Learning.

Provides simulation interfaces for secure aggregation protocols
used in privacy-preserving federated learning.

Mathematical formulation for secret sharing:

$$
s_i = r_{i,1} + r_{i,2} + ... + r_{i,K} \\mod p
$$

where each client $i$ generates random shares for all other clients.

Author: Olaf Yunus Laitinen Imanov <oyli@dtu.dk>
License: EUPL-1.2
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import torch


@dataclass
class SecureAggregationConfig:
    """Configuration for secure aggregation.

    Attributes:
        num_clients: Total number of participating clients.
        threshold: Minimum clients required for reconstruction.
        modulus: Modular arithmetic field size.
    """

    num_clients: int
    threshold: int
    modulus: int = 2**31 - 1


class SecretShare:
    """Additive secret sharing for secure aggregation.

    Splits a value into multiple shares such that the original
    value can only be reconstructed when all shares are combined.

    Args:
        num_shares: Number of shares to create.
        modulus: Modular arithmetic field size.

    Example:
        >>> sharer = SecretShare(num_shares=3, modulus=2**31)
        >>> shares = sharer.split(value)
        >>> reconstructed = sharer.reconstruct(shares)
    """

    def __init__(self, num_shares: int, modulus: int = 2**31 - 1) -> None:
        """Initialize secret sharer.

        Args:
            num_shares: Number of shares to create.
            modulus: Modular arithmetic field size.
        """
        self.num_shares = num_shares
        self.modulus = modulus

    def split(self, value: torch.Tensor) -> List[torch.Tensor]:
        """Split value into additive shares.

        Args:
            value: Tensor to split.

        Returns:
            List of shares that sum to the original value.
        """
        # Convert to long for modular arithmetic
        value_long = (value * 1e6).long() % self.modulus

        shares = []
        remaining = value_long.clone()

        for _ in range(self.num_shares - 1):
            share = torch.randint(0, self.modulus, value_long.shape, dtype=torch.long)
            shares.append(share)
            remaining = (remaining - share) % self.modulus

        shares.append(remaining)
        return shares

    def reconstruct(self, shares: List[torch.Tensor]) -> torch.Tensor:
        """Reconstruct value from shares.

        Args:
            shares: List of additive shares.

        Returns:
            Reconstructed original value.
        """
        total = torch.zeros_like(shares[0])
        for share in shares:
            total = (total + share) % self.modulus

        # Convert back from fixed-point
        return total.float() / 1e6


class SecureAggregation:
    """Secure aggregation protocol simulation.

    Simulates secure aggregation where the server only learns
    the aggregate of client inputs, not individual values.

    Note: This is a simulation for overhead measurement.
    It does not implement actual cryptographic primitives.

    Args:
        config: Secure aggregation configuration.

    Example:
        >>> config = SecureAggregationConfig(num_clients=10, threshold=6)
        >>> sec_agg = SecureAggregation(config)
        >>> aggregate = sec_agg.aggregate(client_updates)
    """

    def __init__(self, config: SecureAggregationConfig) -> None:
        """Initialize secure aggregation.

        Args:
            config: Protocol configuration.
        """
        self.config = config
        self.communication_overhead = 0.0
        self.computation_overhead = 0.0

    def aggregate(
        self,
        inputs: List[Dict[str, torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:
        """Securely aggregate client inputs.

        Args:
            inputs: List of client model updates (state dictionaries).

        Returns:
            Aggregated model update.

        Raises:
            ValueError: If fewer than threshold clients participate.
        """
        if len(inputs) < self.config.threshold:
            raise ValueError(
                f"Need at least {self.config.threshold} clients, got {len(inputs)}"
            )

        # Simulate communication overhead (O(n^2) pairwise shares)
        self.communication_overhead = len(inputs) * (len(inputs) - 1)

        # Simulate computation overhead
        self.computation_overhead = len(inputs) * 2  # Share + reconstruct

        # In simulation, we just compute the sum directly
        result: Dict[str, torch.Tensor] = {}
        first = inputs[0]

        for key in first.keys():
            if isinstance(first[key], torch.Tensor):
                stacked = torch.stack([inp[key] for inp in inputs])
                result[key] = stacked.sum(dim=0)
            else:
                result[key] = first[key]

        return result

    def get_overhead_metrics(self) -> Dict[str, float]:
        """Get overhead metrics from last aggregation.

        Returns:
            Dictionary with communication and computation overhead.
        """
        return {
            "communication_rounds": self.communication_overhead,
            "computation_units": self.computation_overhead,
        }


def simulate_dropout_resilience(
    num_clients: int,
    threshold: int,
    dropout_rate: float,
    num_trials: int = 100,
) -> Dict[str, float]:
    """Simulate dropout resilience of secure aggregation.

    Args:
        num_clients: Total number of clients.
        threshold: Minimum required for reconstruction.
        dropout_rate: Probability of client dropout.
        num_trials: Number of simulation trials.

    Returns:
        Dictionary with success rate and average participants.
    """
    successes = 0
    total_participants = 0
    rng = np.random.default_rng(42)

    for _ in range(num_trials):
        participating = rng.random(num_clients) > dropout_rate
        num_participating = participating.sum()
        total_participants += num_participating

        if num_participating >= threshold:
            successes += 1

    return {
        "success_rate": successes / num_trials,
        "avg_participants": total_participants / num_trials,
        "threshold": float(threshold),
        "num_clients": float(num_clients),
    }
