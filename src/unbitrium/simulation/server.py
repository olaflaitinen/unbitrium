"""Server simulation for federated learning.

Provides the Server class for orchestrating federated training.

Author: Olaf Yunus Laitinen Imanov <oyli@dtu.dk>
License: EUPL-1.2
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from unbitrium.aggregators.base import Aggregator
from unbitrium.aggregators.fedavg import FedAvg


class Server:
    """Federated learning server for orchestrating training.

    Args:
        model_fn: Factory function to create a model instance.
        aggregator: Aggregation algorithm (default: FedAvg).

    Example:
        >>> server = Server(model_fn=lambda: SimpleModel())
        >>> global_state = server.get_global_state()
        >>> server.aggregate(client_updates)
    """

    def __init__(
        self,
        model_fn: Any,
        aggregator: Aggregator | None = None,
    ) -> None:
        """Initialize server.

        Args:
            model_fn: Model factory function.
            aggregator: Aggregation algorithm.
        """
        self.model_fn = model_fn
        self.aggregator = aggregator or FedAvg()
        self.global_model = model_fn()
        self.round_num = 0
        self.history: list[dict[str, float]] = []

    def get_global_state(self) -> dict[str, torch.Tensor]:
        """Get current global model state.

        Returns:
            Model state dictionary.
        """
        return {k: v.clone() for k, v in self.global_model.state_dict().items()}

    def aggregate(
        self,
        updates: list[dict[str, Any]],
    ) -> dict[str, float]:
        """Aggregate client updates.

        Args:
            updates: List of client updates.

        Returns:
            Aggregation metrics.
        """
        self.round_num += 1
        self.global_model, metrics = self.aggregator.aggregate(
            updates, self.global_model
        )
        metrics["round"] = float(self.round_num)
        self.history.append(metrics)
        return metrics

    def evaluate(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
    ) -> dict[str, float]:
        """Evaluate global model.

        Args:
            X: Features.
            y: Labels.

        Returns:
            Evaluation metrics.
        """
        self.global_model.eval()
        with torch.no_grad():
            outputs = self.global_model(X)
            loss = nn.CrossEntropyLoss()(outputs, y).item()
            preds = outputs.argmax(dim=1)
            accuracy = (preds == y).float().mean().item()

        return {"accuracy": accuracy, "loss": loss}
