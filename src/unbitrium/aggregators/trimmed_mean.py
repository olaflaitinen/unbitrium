"""
Trimmed Mean Aggregator.
"""

from typing import Any, Dict, List, Tuple
import torch
from unbitrium.aggregators.base import Aggregator

class TrimmedMean(Aggregator):
    """
    Coordinate-wise Trimmed Mean.

    Robust against Byzantine attackers (outliers).
    Removes the largest and smallest `beta` fraction of values for EACH coordinate.
    """

    def __init__(self, beta: float = 0.1):
        """
        Args:
           beta: Fraction to trim from each end. (0 < beta < 0.5)
        """
        self.beta = beta

    def aggregate(
        self,
        updates: List[Dict[str, Any]],
        current_global_model: torch.nn.Module
    ) -> Tuple[torch.nn.Module, Dict[str, float]]:

        if not updates:
            return current_global_model, {}

        num_clients = len(updates)
        to_remove = int(self.beta * num_clients)

        # If removing too many, fallback to Mean
        if num_clients - 2 * to_remove <= 0:
             # Fallback
             to_remove = 0

        metrics = {
            "trimmed_per_side": to_remove,
            "original_n": num_clients,
            "effective_n": num_clients - 2 * to_remove
        }

        new_sd = {}
        first_state = updates[0]["state_dict"]

        for k in first_state.keys():
            if isinstance(first_state[k], torch.Tensor):
                # Stack all updates for this key: Shape (N, *Shape)
                # Note: This is memory intensive for large models
                stacked = torch.stack([u["state_dict"][k] for u in updates])

                # Sort along batch dim (0)
                sorted_vals, _ = torch.sort(stacked, dim=0)

                # Slice
                start_idx = to_remove
                end_idx = num_clients - to_remove

                trimmed = sorted_vals[start_idx:end_idx]

                # Mean
                mean_val = torch.mean(trimmed, dim=0)
                new_sd[k] = mean_val
            else:
                new_sd[k] = first_state[k]

        current_global_model.load_state_dict(new_sd)
        return current_global_model, metrics
