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


class FederatedSimulator:
    """High-level federated learning simulator.

    Provides a convenient interface for federated simulation with
    pre-split client datasets.

    Args:
        model: Initial model.
        train_datasets: Dictionary mapping client IDs to datasets.
        test_dataset: Test dataset for evaluation.
        aggregator: Aggregation algorithm.
        num_rounds: Number of training rounds.
        clients_per_round: Number of clients per round.
        epochs_per_round: Local epochs per round.
        network_config: Optional network configuration.
    """

    def __init__(
        self,
        model: nn.Module,
        train_datasets: dict[int, Any],
        test_dataset: Any,
        aggregator: Aggregator,
        num_rounds: int = 10,
        clients_per_round: int = 5,
        epochs_per_round: int = 1,
        network_config: Any = None,
        batch_size: int = 32,
        learning_rate: float = 0.01,
        seed: int = 42,
    ) -> None:
        """Initialize federated simulator."""
        self.model = model
        self.train_datasets = train_datasets
        self.test_dataset = test_dataset
        self.aggregator = aggregator
        self.num_rounds = num_rounds
        self.clients_per_round = clients_per_round
        self.epochs_per_round = epochs_per_round
        self.network_config = network_config
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def run(self) -> list[dict[str, float]]:
        """Run federated simulation.

        Returns:
            List of dictionaries with metrics per round.
        """
        history: list[dict[str, float]] = []
        client_ids = list(self.train_datasets.keys())
        global_state = {k: v.clone() for k, v in self.model.state_dict().items()}

        for round_idx in range(self.num_rounds):
            # Select clients
            num_selected = min(self.clients_per_round, len(client_ids))
            selected = self.rng.choice(client_ids, size=num_selected, replace=False)

            # Train selected clients
            updates = []
            for client_id in selected:
                dataset = self.train_datasets[client_id]
                local_model = type(self.model)()
                local_model.load_state_dict(global_state)

                # Local training
                optimizer = torch.optim.SGD(local_model.parameters(), lr=self.learning_rate)
                criterion = nn.CrossEntropyLoss()
                local_model.train()

                for _ in range(self.epochs_per_round):
                    for i in range(0, len(dataset), self.batch_size):
                        end_idx = min(i + self.batch_size, len(dataset))
                        batch = [dataset[j] for j in range(i, end_idx)]
                        x = torch.stack([item[0] for item in batch])
                        y = torch.stack([item[1] for item in batch])
                        optimizer.zero_grad()
                        loss = criterion(local_model(x), y)
                        loss.backward()
                        optimizer.step()

                updates.append({
                    "client_id": client_id,
                    "state_dict": {k: v.clone() for k, v in local_model.state_dict().items()},
                    "num_samples": len(dataset),
                })

            # Aggregate updates
            global_state = self.aggregator.aggregate(updates)
            self.model.load_state_dict(global_state)

            # Evaluate
            self.model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for i in range(len(self.test_dataset)):
                    x, y = self.test_dataset[i]
                    pred = self.model(x.unsqueeze(0)).argmax(dim=1)
                    correct += (pred == y).sum().item()
                    total += 1

            test_acc = correct / total if total > 0 else 0.0
            history.append({
                "round": round_idx,
                "test_acc": test_acc,
                "num_participants": num_selected,
            })

        return history

