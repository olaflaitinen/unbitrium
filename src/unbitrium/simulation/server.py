"""
Federated Server.
"""

from typing import List, Dict
import torch
from unbitrium.aggregators.base import Aggregator
from unbitrium.aggregators.fedavg import FedAvg

class Server:
    """
    Federated Server.
    Orchestrates round, selects clients, aggregates updates.
    """

    def __init__(
        self,
        global_model: torch.nn.Module,
        aggregator: Aggregator = None
    ):
        self.global_model = global_model
        self.aggregator = aggregator or FedAvg()
        self.round_idx = 0

    def select_clients(
        self,
        num_total_clients: int,
        num_to_select: int,
        rng_seed: int = None
    ) -> List[int]:
        """
        Random Selection. Can be overridden for advanced strategies.
        """
        import numpy as np
        rng = np.random.default_rng(rng_seed)
        selected = rng.choice(num_total_clients, size=num_to_select, replace=False)
        return selected.tolist()

    def aggregate_updates(
        self,
        updates: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """
        Calls aggregator.
        """
        if not updates:
            return {}

        self.global_model, metrics = self.aggregator.aggregate(updates, self.global_model)
        self.round_idx += 1
        return metrics

    def get_model(self) -> torch.nn.Module:
        return self.global_model
