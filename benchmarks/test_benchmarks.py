"""Benchmark tests for Unbitrium.

These benchmarks measure the performance of key federated learning operations.
Run with: pytest benchmarks/ --benchmark-only -v

Author: Olaf Yunus Laitinen Imanov <oyli@dtu.dk>
License: EUPL-1.2
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn


class SimpleModel(nn.Module):
    """Simple model for benchmarking."""

    def __init__(self, input_dim: int = 784, hidden_dim: int = 128, num_classes: int = 10):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


def create_mock_updates(num_clients: int, input_dim: int = 784) -> list[dict]:
    """Create mock client updates for benchmarking."""
    model = SimpleModel(input_dim=input_dim)
    updates = []
    for i in range(num_clients):
        state_dict = {k: v.clone() + torch.randn_like(v) * 0.01 for k, v in model.state_dict().items()}
        updates.append({
            "client_id": i,
            "state_dict": state_dict,
            "num_samples": np.random.randint(100, 1000),
        })
    return updates


@pytest.fixture
def simple_model():
    """Create a simple model for benchmarking."""
    return SimpleModel()


@pytest.fixture
def mock_updates_10():
    """Create 10 mock client updates."""
    return create_mock_updates(10)


@pytest.fixture
def mock_updates_100():
    """Create 100 mock client updates."""
    return create_mock_updates(100)


class TestAggregationBenchmarks:
    """Benchmarks for aggregation algorithms."""

    def test_fedavg_aggregation_10_clients(self, benchmark, simple_model, mock_updates_10):
        """Benchmark FedAvg with 10 clients."""
        from unbitrium.aggregators.fedavg import FedAvg

        aggregator = FedAvg()

        def run_aggregation():
            return aggregator.aggregate(mock_updates_10, simple_model)

        result = benchmark(run_aggregation)
        assert result is not None

    def test_fedavg_aggregation_100_clients(self, benchmark, simple_model, mock_updates_100):
        """Benchmark FedAvg with 100 clients."""
        from unbitrium.aggregators.fedavg import FedAvg

        aggregator = FedAvg()

        def run_aggregation():
            return aggregator.aggregate(mock_updates_100, simple_model)

        result = benchmark(run_aggregation)
        assert result is not None


class TestPartitioningBenchmarks:
    """Benchmarks for data partitioning strategies."""

    def test_dirichlet_partitioning(self, benchmark):
        """Benchmark Dirichlet partitioning."""
        from unbitrium.partitioning.dirichlet import DirichletPartitioner

        labels = np.random.randint(0, 10, size=10000)
        partitioner = DirichletPartitioner(num_clients=10, alpha=0.5)

        def run_partition():
            return partitioner.partition(labels)

        result = benchmark(run_partition)
        assert result is not None
        assert len(result) == 10


class TestModelBenchmarks:
    """Benchmarks for model operations."""

    def test_model_forward_pass(self, benchmark, simple_model):
        """Benchmark model forward pass."""
        batch = torch.randn(64, 784)

        def run_forward():
            return simple_model(batch)

        result = benchmark(run_forward)
        assert result is not None
        assert result.shape == (64, 10)

    def test_model_backward_pass(self, benchmark, simple_model):
        """Benchmark model backward pass."""
        batch = torch.randn(64, 784)
        targets = torch.randint(0, 10, (64,))
        criterion = nn.CrossEntropyLoss()

        def run_backward():
            simple_model.zero_grad()
            output = simple_model(batch)
            loss = criterion(output, targets)
            loss.backward()
            return loss

        result = benchmark(run_backward)
        assert result is not None
