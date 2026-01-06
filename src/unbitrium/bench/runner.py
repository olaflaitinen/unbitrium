"""Benchmark runner for Unbitrium.

Orchestrates federated learning experiments with provenance tracking.

Author: Olaf Yunus Laitinen Imanov <oyli@dtu.dk>
License: EUPL-1.2
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from unbitrium.aggregators import FedAvg, FedProx, FedSim
from unbitrium.bench.config import BenchmarkConfig
from unbitrium.core.utils import set_global_seed
from unbitrium.partitioning import DirichletPartitioner


class BenchmarkRunner:
    """Runner for standardized federated learning benchmarks.

    Args:
        config: Benchmark configuration.

    Example:
        >>> config = BenchmarkConfig(name="test", num_rounds=10)
        >>> runner = BenchmarkRunner(config)
        >>> results = runner.run()
    """

    def __init__(self, config: BenchmarkConfig) -> None:
        """Initialize benchmark runner.

        Args:
            config: Benchmark configuration.
        """
        self.config = config
        self._aggregator_registry = {
            "fedavg": FedAvg,
            "fedprox": lambda: FedProx(mu=0.01),
            "fedsim": lambda: FedSim(similarity_threshold=0.5),
        }

    def run(self) -> dict[str, dict[str, list[float]]]:
        """Execute benchmark.

        Returns:
            Dictionary mapping aggregator names to metric histories.
        """
        set_global_seed(self.config.seed)

        # Generate synthetic data
        X, y = self._generate_data()
        num_classes = int(y.max().item()) + 1

        # Partition data
        partitioner = DirichletPartitioner(
            num_clients=self.config.num_clients,
            alpha=self.config.partitioning.get("alpha", 0.5),
            seed=self.config.seed,
        )
        client_indices = partitioner.partition(y.numpy())

        # Run for each aggregator
        results: dict[str, dict[str, list[float]]] = {}

        for agg_name in self.config.aggregators:
            if agg_name.lower() not in self._aggregator_registry:
                continue

            agg_factory = self._aggregator_registry[agg_name.lower()]
            aggregator = agg_factory() if callable(agg_factory) else agg_factory

            history = self._run_experiment(
                X, y, num_classes, client_indices, aggregator
            )
            results[agg_name] = history

        return results

    def _generate_data(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate synthetic classification data."""
        num_samples = 1000 * self.config.num_clients
        num_features = 20
        num_classes = 10

        X = torch.randn(num_samples, num_features)
        y = torch.randint(0, num_classes, (num_samples,))
        return X, y

    def _run_experiment(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        num_classes: int,
        client_indices: dict[int, list[int]],
        aggregator: Any,
    ) -> dict[str, list[float]]:
        """Run single experiment with one aggregator."""
        model = self._create_model(X.shape[1], num_classes)
        history: dict[str, list[float]] = {"accuracy": [], "loss": []}

        for _ in range(self.config.num_rounds):
            updates = []

            for client_id, indices in client_indices.items():
                if len(indices) == 0:
                    continue

                local_X = X[indices]
                local_y = y[indices]

                local_model = self._create_model(X.shape[1], num_classes)
                local_model.load_state_dict(model.state_dict())

                state_dict, num_samples = self._train_local(
                    local_model, local_X, local_y
                )
                updates.append({"state_dict": state_dict, "num_samples": num_samples})

            if updates:
                model, _ = aggregator.aggregate(updates, model)

            # Evaluate
            acc, loss = self._evaluate(model, X, y)
            history["accuracy"].append(acc)
            history["loss"].append(loss)

        return history

    def _create_model(self, input_dim: int, num_classes: int) -> nn.Module:
        """Create simple neural network model."""
        return nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def _train_local(
        self,
        model: nn.Module,
        X: torch.Tensor,
        y: torch.Tensor,
    ) -> tuple[dict[str, torch.Tensor], int]:
        """Train model on local data."""
        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True)

        optimizer = torch.optim.SGD(model.parameters(), lr=self.config.learning_rate)
        criterion = nn.CrossEntropyLoss()

        model.train()
        for _ in range(self.config.local_epochs):
            for X_batch, y_batch in loader:
                optimizer.zero_grad()
                loss = criterion(model(X_batch), y_batch)
                loss.backward()
                optimizer.step()

        return {k: v.clone() for k, v in model.state_dict().items()}, len(X)

    def _evaluate(
        self,
        model: nn.Module,
        X: torch.Tensor,
        y: torch.Tensor,
    ) -> tuple[float, float]:
        """Evaluate model."""
        model.eval()
        with torch.no_grad():
            outputs = model(X)
            loss = nn.CrossEntropyLoss()(outputs, y).item()
            preds = outputs.argmax(dim=1)
            acc = (preds == y).float().mean().item()
        return acc, loss
