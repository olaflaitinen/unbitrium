"""TrimmedMean Aggregator implementation.

Coordinate-wise trimmed mean for robust aggregation.

Mathematical formulation:

$$
\\text{TM}(\\{x_k\\}) = \\frac{1}{K - 2b} \\sum_{k=b+1}^{K-b} x_{(k)}
$$

where $x_{(k)}$ is the $k$-th smallest value and $b$ is the trim fraction.

Author: Olaf Yunus Laitinen Imanov <oyli@dtu.dk>
License: EUPL-1.2
"""

from __future__ import annotations

from typing import Any

import torch

from unbitrium.aggregators.base import Aggregator


class TrimmedMean(Aggregator):
    """Trimmed mean robust aggregator.

    Computes coordinate-wise trimmed mean by excluding extreme values,
    providing robustness against outliers and Byzantine failures.

    Args:
        trim_ratio: Fraction of values to trim from each end (0 to 0.5).

    Example:
        >>> aggregator = TrimmedMean(trim_ratio=0.1)
        >>> new_model, metrics = aggregator.aggregate(updates, global_model)
    """

    def __init__(self, trim_ratio: float = 0.1) -> None:
        """Initialize TrimmedMean aggregator.

        Args:
            trim_ratio: Fraction to trim from each end.
        """
        if not 0 <= trim_ratio < 0.5:
            raise ValueError("trim_ratio must be in [0, 0.5)")
        self.trim_ratio = trim_ratio

    def aggregate(
        self,
        updates: list[dict[str, Any]],
        current_global_model: torch.nn.Module,
    ) -> tuple[torch.nn.Module, dict[str, float]]:
        """Aggregate using coordinate-wise trimmed mean.

        Args:
            updates: List of client updates with 'state_dict' and 'num_samples'.
            current_global_model: Current global model.

        Returns:
            Tuple of (updated global model, aggregation metrics).
        """
        if not updates:
            return current_global_model, {"aggregated_clients": 0.0}

        num_clients = len(updates)
        trim_count = int(num_clients * self.trim_ratio)

        new_state_dict: dict[str, torch.Tensor] = {}
        first_state = updates[0]["state_dict"]

        for key in first_state.keys():
            if isinstance(first_state[key], torch.Tensor):
                # Stack all client values for this parameter
                stacked = torch.stack([
                    u["state_dict"][key].float() for u in updates
                ], dim=0)

                if trim_count > 0 and num_clients > 2 * trim_count:
                    # Sort along client dimension
                    sorted_vals, _ = torch.sort(stacked, dim=0)
                    # Trim extremes
                    trimmed = sorted_vals[trim_count : num_clients - trim_count]
                    # Mean of remaining
                    result = trimmed.mean(dim=0)
                else:
                    # Not enough clients to trim
                    result = stacked.mean(dim=0)

                new_state_dict[key] = result.to(first_state[key].dtype)
            else:
                new_state_dict[key] = first_state[key]

        current_global_model.load_state_dict(new_state_dict)

        metrics = {
            "num_participants": float(num_clients),
            "trim_count": float(trim_count),
            "effective_clients": float(max(1, num_clients - 2 * trim_count)),
        }
        return current_global_model, metrics
