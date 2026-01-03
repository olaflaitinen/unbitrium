"""
FedProx Aggregator.
"""

from typing import Any, Dict, List, Tuple
import torch
from unbitrium.aggregators.base import Aggregator

class FedProx(Aggregator):
    """
    FedProx Aggregator.

    Note: FedProx logic primarily affects the CLIENT local training (adding proximal term).
    The server-side aggregation is typically identical to FedAvg (weighted average).
    However, some variants normalize differently. We implement standard weighted average here.

    Mathematical formulation (Client Objective):
    $$
    \min_w F_k(w) + \frac{\mu}{2}\|w - w^t\|_2^2
    $$

    Server Aggregation:
    $$
    w^{t+1} = \sum_k \frac{n_k}{N} w_k^{t+1}
    $$
    """

    def __init__(self, mu: float = 0.01):
        self.mu = mu

    def aggregate(
        self,
        updates: List[Dict[str, Any]],
        current_global_model: torch.nn.Module
    ) -> Tuple[torch.nn.Module, Dict[str, float]]:
        # FedProx server side is identical to FedAvg
        # We duplicate logic or inherit, but explicit is better for the repository

        if not updates:
            return current_global_model, {"aggregated_clients": 0.0}

        total_samples = sum(u["num_samples"] for u in updates)
        if total_samples == 0:
             return current_global_model, {"aggregated_clients": 0.0}

        new_state_dict = {}
        first_state = updates[0]["state_dict"]

        for key in first_state.keys():
            if isinstance(first_state[key], torch.Tensor):
                weighted_sum = torch.zeros_like(first_state[key], dtype=first_state[key].dtype)
                for update in updates:
                    weight = update["num_samples"] / total_samples
                    # Ensure dtype match
                    val = update["state_dict"][key]
                    if val.dtype != weighted_sum.dtype:
                         val = val.to(weighted_sum.dtype)
                    weighted_sum += val * weight
                new_state_dict[key] = weighted_sum
            else:
                new_state_dict[key] = first_state[key]

        current_global_model.load_state_dict(new_state_dict)

        metrics = {
            "num_participants": len(updates),
            "total_samples": float(total_samples),
            "prox_mu": self.mu
        }
        return current_global_model, metrics
