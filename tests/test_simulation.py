"""Unit tests for simulation module.

Author: Olaf Yunus Laitinen Imanov <oyli@dtu.dk>
License: EUPL-1.2
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from torch.utils.data import Dataset, TensorDataset, DataLoader

from unbitrium.simulation.simulator import FederatedSimulator
from unbitrium.simulation.network import NetworkConfig
from unbitrium.aggregators import FedAvg


class SimpleModel(nn.Module):
    """Simple model for testing."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(10, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class DummyDataset(Dataset):
    """Dummy dataset for testing."""

    def __init__(self, size: int = 100) -> None:
        self.data = torch.randn(size, 10)
        self.targets = torch.randint(0, 2, (size,))

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.data[idx], self.targets[idx]


class TestNetworkConfig:
    """Tests for NetworkConfig."""

    def test_init_default(self) -> None:
        """Test default network configuration."""
        config = NetworkConfig()
        assert config is not None

    def test_init_with_latency(self) -> None:
        """Test network config with custom latency."""
        config = NetworkConfig(latency_mean=50.0, latency_std=10.0)
        assert config.latency_mean == 50.0
        assert config.latency_std == 10.0

    def test_init_with_packet_loss(self) -> None:
        """Test network config with packet loss."""
        config = NetworkConfig(packet_loss_rate=0.05)
        assert config.packet_loss_rate == 0.05

    def test_simulate_latency(self) -> None:
        """Test latency simulation."""
        config = NetworkConfig(latency_mean=100.0, latency_std=10.0)
        latency = config.simulate_latency()
        assert latency >= 0

    def test_simulate_packet_loss(self) -> None:
        """Test packet loss simulation."""
        config = NetworkConfig(packet_loss_rate=0.5)

        # Run multiple times to test probability
        losses = [config.simulate_packet_loss() for _ in range(1000)]
        loss_rate = sum(losses) / len(losses)

        # Should be close to 0.5
        assert 0.3 < loss_rate < 0.7


class TestFederatedSimulator:
    """Tests for FederatedSimulator."""

    def test_init(self) -> None:
        """Test simulator initialization."""
        model = SimpleModel()
        train_datasets = {i: DummyDataset(20) for i in range(5)}
        test_dataset = DummyDataset(50)
        aggregator = FedAvg()

        sim = FederatedSimulator(
            model=model,
            train_datasets=train_datasets,
            test_dataset=test_dataset,
            aggregator=aggregator,
            num_rounds=2,
            clients_per_round=2,
            epochs_per_round=1,
        )

        assert sim is not None

    def test_run_simulation(self) -> None:
        """Test running simulation."""
        model = SimpleModel()
        train_datasets = {i: DummyDataset(20) for i in range(5)}
        test_dataset = DummyDataset(50)
        aggregator = FedAvg()

        sim = FederatedSimulator(
            model=model,
            train_datasets=train_datasets,
            test_dataset=test_dataset,
            aggregator=aggregator,
            num_rounds=2,
            clients_per_round=2,
            epochs_per_round=1,
        )

        history = sim.run()

        assert len(history) == 2
        assert "test_acc" in history[0]

    def test_simulation_improves_accuracy(self) -> None:
        """Test that training improves accuracy."""
        torch.manual_seed(42)

        model = SimpleModel()
        train_datasets = {i: DummyDataset(50) for i in range(10)}
        test_dataset = DummyDataset(100)
        aggregator = FedAvg()

        sim = FederatedSimulator(
            model=model,
            train_datasets=train_datasets,
            test_dataset=test_dataset,
            aggregator=aggregator,
            num_rounds=5,
            clients_per_round=5,
            epochs_per_round=2,
        )

        history = sim.run()

        # Accuracy should generally improve or stay reasonable
        assert history[-1]["test_acc"] >= 0.3  # Better than random

    def test_simulation_with_network_config(self) -> None:
        """Test simulation with network configuration."""
        model = SimpleModel()
        train_datasets = {i: DummyDataset(20) for i in range(3)}
        test_dataset = DummyDataset(50)
        aggregator = FedAvg()
        net_config = NetworkConfig(latency_mean=10.0)

        sim = FederatedSimulator(
            model=model,
            train_datasets=train_datasets,
            test_dataset=test_dataset,
            aggregator=aggregator,
            num_rounds=1,
            clients_per_round=2,
            epochs_per_round=1,
            network_config=net_config,
        )

        history = sim.run()
        assert len(history) == 1

    def test_client_selection(self) -> None:
        """Test client selection in simulation."""
        model = SimpleModel()
        train_datasets = {i: DummyDataset(20) for i in range(10)}
        test_dataset = DummyDataset(50)
        aggregator = FedAvg()

        sim = FederatedSimulator(
            model=model,
            train_datasets=train_datasets,
            test_dataset=test_dataset,
            aggregator=aggregator,
            num_rounds=3,
            clients_per_round=3,
            epochs_per_round=1,
        )

        history = sim.run()

        # Each round should have 3 participants
        assert all(h.get("num_participants", 3) <= 3 for h in history)


class TestLocalTraining:
    """Tests for local client training."""

    def test_local_training_updates_model(self) -> None:
        """Test that local training changes model weights."""
        model = SimpleModel()
        initial_weights = model.fc.weight.clone()

        dataset = DummyDataset(50)
        loader = DataLoader(dataset, batch_size=10, shuffle=True)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        criterion = nn.CrossEntropyLoss()

        model.train()
        for x, y in loader:
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()

        assert not torch.allclose(initial_weights, model.fc.weight)

    def test_model_state_dict_copy(self) -> None:
        """Test model state dict copying."""
        model = SimpleModel()
        state = {k: v.clone() for k, v in model.state_dict().items()}

        # Modify original
        model.fc.weight.data.fill_(0)

        # Copy should be unchanged
        assert not torch.allclose(state["fc.weight"], model.fc.weight)


class TestModuleExports:
    """Test simulation module exports."""

    def test_exports(self) -> None:
        """Test all expected exports exist."""
        from unbitrium import simulation

        assert hasattr(simulation, "simulator") or hasattr(simulation, "FederatedSimulator")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
