
import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from unbitrium.aggregators.fedavg import FedAvg
from unbitrium.partitioning.quantity_skew import QuantitySkewPowerLaw
from unbitrium.simulation.simulator import FederatedSimulator
from unbitrium.simulation.network import NetworkConfig

# Dummy Dataset
class DummyDataset(Dataset):
    def __init__(self, size=100):
        self.size = size
        self.data = torch.randn(size, 10)
        self.targets = torch.randint(0, 2, (size,))

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

# Dummy Model
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 2)

    def forward(self, x):
        return self.fc(x)

def test_smoke_simulation():
    print("Running Smoke Test...")

    # Setup
    num_clients = 5
    dataset = DummyDataset(size=200)
    test_dataset = DummyDataset(size=50)

    # Partition
    partitioner = QuantitySkewPowerLaw(num_clients=num_clients, gamma=0.5)
    # Mocking partitioners internal _get_targets which expects dataset to have 'targets' or 'labels' or 'y'
    # Our DummyDataset has .targets.
    # But Partitioner Base might need implementation details.
    # Let's bypass Partitioner Base logic if it expects specific standard datasets,
    # but QuantitySkew only needs .targets if implemented robustly...
    # Checking QuantitySkew implementation: "targets = self._get_targets(dataset)"
    # I need to ensure _get_targets works.
    # But since I cannot easily see base.py, I will manually partition for this smoke test to isolate Simulator.

    # Manual Partitioning
    indices = {i: list(range(i*20, (i+1)*20)) for i in range(num_clients)}
    client_datasets = {i: torch.utils.data.Subset(dataset, idxs) for i, idxs in indices.items()}

    model = SimpleModel()
    aggregator = FedAvg()
    net_config = NetworkConfig(latency_mean=0.0)

    sim = FederatedSimulator(
        model=model,
        train_datasets=client_datasets,
        test_dataset=test_dataset,
        aggregator=aggregator,
        num_rounds=2,
        clients_per_round=2,
        epochs_per_round=1,
        network_config=net_config
    )

    history = sim.run()

    assert len(history) == 2
    assert "test_acc" in history[0]
    print("Smoke Test Passed!")

if __name__ == "__main__":
    test_smoke_simulation()
