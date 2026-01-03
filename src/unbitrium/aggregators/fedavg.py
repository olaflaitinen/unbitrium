"""
FedAvg Aggregator.
"""

from typing import Any, Dict, List, Tuple
import torch
from unbitrium.aggregators.base import Aggregator

class FedAvg(Aggregator):
    """
    Federated Averaging (FedAvg).

    Mathematical formulation:
    $$
    w^{t+1} = \sum_{k=1}^K \frac{n_k}{N} w_k^t
    $$
    where $n_k$ is the number of samples on client $k$, and $N = \sum n_k$.
    """

    def aggregate(
        self,
        updates: List[Dict[str, Any]],
        current_global_model: torch.nn.Module
    ) -> Tuple[torch.nn.Module, Dict[str, float]]:
        """
        Aggregates model updates using weighted average.

        Args:
            updates: List of dictionaries containing 'state_dict' and 'num_samples'.
            current_global_model: The global model at the current round (used for structure).

        Returns:
            Updated global model and metrics.
        """
        if not updates:
            return current_global_model, {"aggregated_clients": 0.0}

        total_samples = sum(u["num_samples"] for u in updates)
        if total_samples == 0:
             return current_global_model, {"aggregated_clients": 0.0}

        # Initialize averaged state dict
        new_state_dict = {}
        first_state = updates[0]["state_dict"]

        for key in first_state.keys():
            # Check dtype to handle non-tensor buffers if any (usually float/int)
            # We assume torch tensors for weights
            if isinstance(first_state[key], torch.Tensor):
                # Vectorized sum if possible, but here we loop clients
                weighted_sum = torch.zeros_like(first_state[key])
                for update in updates:
                    weight = update["num_samples"] / total_samples
                    weighted_sum += update["state_dict"][key] * weight
                new_state_dict[key] = weighted_sum
            else:
                # discrete values or metadata, take from first or largest?
                # For safety in FedAvg, we usually take the base.
                # But strict FedAvg implies averaging weights.
                new_state_dict[key] = first_state[key]

        # In-place update or new instance?
        # Usually safer to load into the existing instance to preserve object identity
        current_global_model.load_state_dict(new_state_dict)

        metrics = {
            "num_participants": len(updates),
            "total_samples": float(total_samples)
        }
        return current_global_model, metrics
