"""Base aggregator interface for Unbitrium.

Defines the abstract base class that all aggregation algorithms must implement.

Author: Olaf Yunus Laitinen Imanov <oyli@dtu.dk>
License: EUPL-1.2
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch


class Aggregator(ABC):
    """Abstract base class for federated learning aggregators.

    All aggregation algorithms (FedAvg, FedProx, FedSim, etc.) must
    inherit from this class and implement the aggregate method.
    """

    @abstractmethod
    def aggregate(
        self,
        updates: list[dict[str, Any]],
        current_global_model: torch.nn.Module,
    ) -> tuple[torch.nn.Module, dict[str, float]]:
        """Aggregate client model updates into a new global model.

        Args:
            updates: List of client updates. Each update is a dictionary
                containing at minimum 'state_dict' and 'num_samples'.
            current_global_model: The current global model.

        Returns:
            Tuple of (updated global model, aggregation metrics).
        """
        pass
