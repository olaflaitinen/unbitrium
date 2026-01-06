"""Benchmark configuration for Unbitrium.

Provides dataclass for standardized experiment configuration.

Author: Olaf Yunus Laitinen Imanov <oyli@dtu.dk>
License: EUPL-1.2
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class BenchmarkConfig:
    """Configuration for federated learning benchmarks.

    Attributes:
        name: Experiment name.
        dataset: Dataset name.
        num_clients: Number of clients.
        num_rounds: Number of federated rounds.
        local_epochs: Local training epochs per round.
        batch_size: Training batch size.
        learning_rate: Learning rate.
        partitioning: Partitioning strategy configuration.
        aggregators: List of aggregator names to compare.
        metrics: List of metrics to track.
        seed: Random seed.
    """

    name: str = "unnamed_benchmark"
    dataset: str = "synthetic"
    num_clients: int = 10
    num_rounds: int = 10
    local_epochs: int = 1
    batch_size: int = 32
    learning_rate: float = 0.01
    partitioning: dict[str, Any] = field(
        default_factory=lambda: {"strategy": "dirichlet", "alpha": 0.5}
    )
    aggregators: list[str] = field(default_factory=lambda: ["fedavg"])
    metrics: list[str] = field(default_factory=lambda: ["accuracy", "loss"])
    seed: int = 42

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary.

        Returns:
            Configuration dictionary.
        """
        return {
            "name": self.name,
            "dataset": self.dataset,
            "num_clients": self.num_clients,
            "num_rounds": self.num_rounds,
            "local_epochs": self.local_epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "partitioning": self.partitioning,
            "aggregators": self.aggregators,
            "metrics": self.metrics,
            "seed": self.seed,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BenchmarkConfig:
        """Create configuration from dictionary.

        Args:
            data: Configuration dictionary.

        Returns:
            BenchmarkConfig instance.
        """
        return cls(
            name=data.get("experiment", {}).get("name", data.get("name", "unnamed")),
            dataset=data.get("dataset", {}).get(
                "name", data.get("dataset", "synthetic")
            ),
            num_clients=data.get("clients", {}).get(
                "num_clients", data.get("num_clients", 10)
            ),
            num_rounds=data.get("training", {}).get(
                "num_rounds", data.get("num_rounds", 10)
            ),
            local_epochs=data.get("training", {}).get(
                "local_epochs", data.get("local_epochs", 1)
            ),
            batch_size=data.get("training", {}).get(
                "batch_size", data.get("batch_size", 32)
            ),
            learning_rate=data.get("training", {}).get(
                "learning_rate", data.get("learning_rate", 0.01)
            ),
            partitioning=data.get(
                "partitioning", {"strategy": "dirichlet", "alpha": 0.5}
            ),
            aggregators=data.get(
                "aggregators", [data.get("aggregator", {}).get("name", "fedavg")]
            ),
            metrics=data.get("metrics", ["accuracy", "loss"]),
            seed=data.get("reproducibility", {}).get("seed", data.get("seed", 42)),
        )
