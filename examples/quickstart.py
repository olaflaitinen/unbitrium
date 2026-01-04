"""Quickstart example for Unbitrium federated learning simulation.

This script demonstrates a minimal end-to-end federated learning workflow
using FedAvg aggregation on a synthetically partitioned dataset.

Author: Olaf Yunus Laitinen Imanov <oyli@dtu.dk>
License: EUPL-1.2
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from unbitrium.aggregators import FedAvg
from unbitrium.partitioning import DirichletPartitioner
from unbitrium.metrics.heterogeneity import compute_label_entropy


def create_synthetic_data(
    num_samples: int = 1000,
    num_features: int = 10,
    num_classes: int = 5,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate synthetic classification data."""
    torch.manual_seed(seed)
    X = torch.randn(num_samples, num_features)
    y = torch.randint(0, num_classes, (num_samples,))
    return X, y


class SimpleModel(nn.Module):
    """Simple feedforward network for demonstration."""

    def __init__(self, input_dim: int, num_classes: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


def train_local(
    model: nn.Module,
    dataloader: DataLoader,
    epochs: int = 1,
    lr: float = 0.01,
) -> dict[str, torch.Tensor]:
    """Train model locally and return state dict."""
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for _ in range(epochs):
        for X_batch, y_batch in dataloader:
            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()

    return {k: v.clone() for k, v in model.state_dict().items()}


def main() -> None:
    """Run federated learning simulation."""
    # Configuration
    num_clients = 10
    num_rounds = 5
    local_epochs = 2
    alpha = 0.5  # Dirichlet concentration (lower = more heterogeneous)

    # Generate data
    X, y = create_synthetic_data(num_samples=2000, num_features=10, num_classes=5)

    # Partition data across clients using Dirichlet distribution
    partitioner = DirichletPartitioner(
        num_clients=num_clients,
        alpha=alpha,
        seed=42,
    )
    client_indices = partitioner.partition(y.numpy())

    # Compute heterogeneity metric
    entropy = compute_label_entropy(y.numpy(), client_indices)
    print(f"Label entropy across clients: {entropy:.4f}")

    # Initialize global model
    global_model = SimpleModel(input_dim=10, num_classes=5)
    aggregator = FedAvg()

    # Federated training loop
    for round_idx in range(num_rounds):
        client_updates = []
        client_weights = []

        for client_id in range(num_clients):
            # Get client data
            indices = client_indices[client_id]
            if len(indices) == 0:
                continue

            client_X = X[indices]
            client_y = y[indices]
            dataset = TensorDataset(client_X, client_y)
            dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

            # Create local model copy
            local_model = SimpleModel(input_dim=10, num_classes=5)
            local_model.load_state_dict(global_model.state_dict())

            # Train locally
            local_state = train_local(local_model, dataloader, epochs=local_epochs)
            client_updates.append(local_state)
            client_weights.append(len(indices))

        # Aggregate updates
        aggregated_state = aggregator.aggregate(client_updates, client_weights)
        global_model.load_state_dict(aggregated_state)

        print(f"Round {round_idx + 1}/{num_rounds} completed")

    print("Federated training complete.")


if __name__ == "__main__":
    main()
