"""Integration tests for Unbitrium.

Author: Olaf Yunus Laitinen Imanov <oyli@dtu.dk>
License: EUPL-1.2
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from unbitrium import (
    FedAvg,
    FedSim,
    DirichletPartitioner,
    compute_label_entropy,
    compute_emd,
    set_global_seed,
    BenchmarkRunner,
    BenchmarkConfig,
)


class SimpleModel(nn.Module):
    """Simple model for integration tests."""

    def __init__(self, input_dim: int = 10, num_classes: int = 5) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 32)
        self.fc2 = nn.Linear(32, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


class TestEndToEndWorkflow:
    """End-to-end integration tests."""

    def test_full_federated_round(self) -> None:
        """Test complete federated learning round."""
        set_global_seed(42)

        # Generate data
        num_samples = 500
        X = torch.randn(num_samples, 10)
        y = torch.randint(0, 5, (num_samples,))

        # Partition
        partitioner = DirichletPartitioner(num_clients=5, alpha=0.5)
        client_indices = partitioner.partition(y.numpy())

        # Compute heterogeneity metrics
        entropy = compute_label_entropy(y.numpy(), client_indices)
        emd = compute_emd(y.numpy(), client_indices)
        assert entropy > 0
        assert emd >= 0

        # Train one round
        global_model = SimpleModel()
        aggregator = FedAvg()

        updates = []
        for client_id, indices in client_indices.items():
            if len(indices) == 0:
                continue

            # Local training
            local_model = SimpleModel()
            local_model.load_state_dict(global_model.state_dict())

            client_X = X[indices]
            client_y = y[indices]
            dataset = TensorDataset(client_X, client_y)
            loader = DataLoader(dataset, batch_size=32, shuffle=True)

            optimizer = torch.optim.SGD(local_model.parameters(), lr=0.01)
            criterion = nn.CrossEntropyLoss()

            local_model.train()
            for X_batch, y_batch in loader:
                optimizer.zero_grad()
                loss = criterion(local_model(X_batch), y_batch)
                loss.backward()
                optimizer.step()

            updates.append({
                "state_dict": {k: v.clone() for k, v in local_model.state_dict().items()},
                "num_samples": len(indices),
            })

        # Aggregate
        new_model, metrics = aggregator.aggregate(updates, global_model)
        assert metrics["num_participants"] == 5

    def test_benchmark_runner(self) -> None:
        """Test benchmark runner."""
        config = BenchmarkConfig(
            name="test_benchmark",
            num_clients=5,
            num_rounds=2,
            local_epochs=1,
        )
        runner = BenchmarkRunner(config)
        results = runner.run()

        assert "fedavg" in results
        assert len(results["fedavg"]["accuracy"]) == 2


class TestAggregatorComparison:
    """Compare different aggregators."""

    def test_fedavg_vs_fedsim(self) -> None:
        """Compare FedAvg and FedSim."""
        set_global_seed(42)

        model = SimpleModel()
        updates = []
        for i in range(5):
            state = {k: v.clone() + torch.randn_like(v) * 0.1 for k, v in model.state_dict().items()}
            updates.append({"state_dict": state, "num_samples": 100})

        # FedAvg
        fedavg = FedAvg()
        model1 = SimpleModel()
        _, metrics1 = fedavg.aggregate(updates, model1)

        # FedSim
        fedsim = FedSim(similarity_threshold=0.0)
        model2 = SimpleModel()
        _, metrics2 = fedsim.aggregate(updates, model2)

        assert metrics1["num_participants"] == metrics2["num_participants"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
