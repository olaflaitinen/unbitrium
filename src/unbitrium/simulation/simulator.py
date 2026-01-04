"""Federated learning simulator.

Provides end-to-end simulation orchestration.

Author: Olaf Yunus Laitinen Imanov <oyli@dtu.dk>
License: EUPL-1.2
"""

from __future__ import annotations

from typing import Any, Callable

import torch
import torch.nn as nn
import numpy as np

from unbitrium.simulation.client import Client
from unbitrium.simulation.server import Server
from unbitrium.simulation.network import Network
from unbitrium.aggregators.base import Aggregator
from unbitrium.aggregators.fedavg import FedAvg
from unbitrium.partitioning.base import Partitioner
from unbitrium.partitioning.dirichlet import DirichletPartitioner


class Simulator:
    """End-to-end federated learning simulator.

    Args:
        model_fn: Factory function to create a model instance.
        dataset: Tuple of (features, labels) tensors.
        num_clients: Number of clients.
        aggregator: Aggregation algorithm.
        partitioner: Data partitioning strategy.
        participation_rate: Fraction of clients per round.
        seed: Random seed.

    Example:
        >>> sim = Simulator(
        ...     model_fn=lambda: SimpleModel(),
        ...     dataset=(X, y),
        ...     num_clients=10,
        ... )
        >>> results = sim.run(num_rounds=10)
    """

    def __init__(
        self,
        model_fn: Callable[[], nn.Module],
        dataset: tuple[torch.Tensor, torch.Tensor],
        num_clients: int = 10,
        aggregator: Aggregator | None = None,
        partitioner: Partitioner | None = None,
        participation_rate: float = 1.0,
        local_epochs: int = 1,
        batch_size: int = 32,
        learning_rate: float = 0.01,
        seed: int = 42,
    ) -> None:
        """Initialize simulator.

        Args:
            model_fn: Model factory function.
            dataset: (features, labels) tuple.
            num_clients: Number of clients.
            aggregator: Aggregation algorithm.
            partitioner: Data partitioner.
            participation_rate: Client participation rate.
            local_epochs: Local training epochs.
            batch_size: Training batch size.
            learning_rate: Local learning rate.
            seed: Random seed.
        """
        self.model_fn = model_fn
        self.dataset = dataset
        self.num_clients = num_clients
        self.aggregator = aggregator or FedAvg()
        self.partitioner = partitioner or DirichletPartitioner(num_clients, alpha=0.5, seed=seed)
        self.participation_rate = participation_rate
        self.local_epochs = local_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        # Set up components
        self._setup()

    def _setup(self) -> None:
        """Set up simulation components."""
        X, y = self.dataset

        # Partition data
        client_indices = self.partitioner.partition(y.numpy())

        # Create clients
        self.clients: list[Client] = []
        for client_id in range(self.num_clients):
            indices = client_indices.get(client_id, [])
            if len(indices) > 0:
                client_X = X[indices]
                client_y = y[indices]
                client = Client(
                    client_id=client_id,
                    local_data=(client_X, client_y),
                    model_fn=self.model_fn,
                    batch_size=self.batch_size,
                    learning_rate=self.learning_rate,
                    local_epochs=self.local_epochs,
                )
                self.clients.append(client)

        # Create server
        self.server = Server(model_fn=self.model_fn, aggregator=self.aggregator)

        # Create network
        self.network = Network(num_clients=len(self.clients), seed=self.seed)

    def run(
        self,
        num_rounds: int = 10,
    ) -> dict[str, list[float]]:
        """Run federated simulation.

        Args:
            num_rounds: Number of training rounds.

        Returns:
            Dictionary with metric histories.
        """
        history: dict[str, list[float]] = {
            "accuracy": [],
            "loss": [],
            "num_participants": [],
        }

        X, y = self.dataset

        for round_idx in range(num_rounds):
            # Select participating clients
            num_selected = max(1, int(len(self.clients) * self.participation_rate))
            selected = self.rng.choice(
                self.clients, size=num_selected, replace=False
            ).tolist()

            # Get global state
            global_state = self.server.get_global_state()

            # Train selected clients
            updates = []
            for client in selected:
                update = client.train(global_state)
                updates.append(update)

            # Aggregate
            metrics = self.server.aggregate(updates)
            history["num_participants"].append(float(len(updates)))

            # Evaluate
            eval_metrics = self.server.evaluate(X, y)
            history["accuracy"].append(eval_metrics["accuracy"])
            history["loss"].append(eval_metrics["loss"])

        return history
