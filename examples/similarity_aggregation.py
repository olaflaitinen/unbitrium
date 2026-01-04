"""Similarity-guided aggregation demonstration.

This script compares FedAvg, FedSim, and pFedSim aggregators under
heterogeneous data distributions.

Author: Olaf Yunus Laitinen Imanov <oyli@dtu.dk>
License: EUPL-1.2
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from unbitrium.aggregators import FedAvg, FedSim, PFedSim
from unbitrium.partitioning import DirichletPartitioner
from unbitrium.metrics.heterogeneity import compute_label_entropy


class MLP(nn.Module):
    """Multi-layer perceptron for classification."""

    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int) -> None:
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.head = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.shared(x)
        return self.head(features)


def create_data(
    num_samples: int,
    num_features: int,
    num_classes: int,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate synthetic classification data with class structure."""
    torch.manual_seed(seed)

    X_list = []
    y_list = []

    samples_per_class = num_samples // num_classes
    for cls in range(num_classes):
        # Each class has a distinct mean
        mean = torch.zeros(num_features)
        mean[cls % num_features] = 2.0
        X_cls = torch.randn(samples_per_class, num_features) + mean
        y_cls = torch.full((samples_per_class,), cls, dtype=torch.long)
        X_list.append(X_cls)
        y_list.append(y_cls)

    return torch.cat(X_list), torch.cat(y_list)


def evaluate(model: nn.Module, X: torch.Tensor, y: torch.Tensor) -> float:
    """Evaluate model accuracy."""
    model.eval()
    with torch.no_grad():
        preds = model(X).argmax(dim=1)
        accuracy = (preds == y).float().mean().item()
    return accuracy


def train_round(
    global_model: nn.Module,
    client_indices: dict[int, list[int]],
    X: torch.Tensor,
    y: torch.Tensor,
    aggregator,
    local_epochs: int = 2,
    lr: float = 0.01,
) -> dict[str, torch.Tensor]:
    """Train one federated round."""
    client_updates = []
    client_weights = []

    for client_id, indices in client_indices.items():
        if len(indices) == 0:
            continue

        # Local data
        client_X = X[indices]
        client_y = y[indices]
        dataset = TensorDataset(client_X, client_y)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

        # Local model
        local_model = MLP(input_dim=X.shape[1], hidden_dim=64, num_classes=y.max().item() + 1)
        local_model.load_state_dict(global_model.state_dict())

        # Train
        optimizer = torch.optim.SGD(local_model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        local_model.train()

        for _ in range(local_epochs):
            for X_batch, y_batch in dataloader:
                optimizer.zero_grad()
                loss = criterion(local_model(X_batch), y_batch)
                loss.backward()
                optimizer.step()

        client_updates.append({k: v.clone() for k, v in local_model.state_dict().items()})
        client_weights.append(len(indices))

    # Aggregate
    return aggregator.aggregate(client_updates, client_weights)


def main() -> None:
    """Compare aggregation strategies."""
    # Configuration
    num_clients = 20
    num_rounds = 10
    alpha = 0.3  # High heterogeneity

    # Data
    X, y = create_data(num_samples=5000, num_features=20, num_classes=10)
    num_classes = y.max().item() + 1

    # Partition
    partitioner = DirichletPartitioner(num_clients=num_clients, alpha=alpha, seed=42)
    client_indices = partitioner.partition(y.numpy())

    entropy = compute_label_entropy(y.numpy(), client_indices)
    print(f"Data heterogeneity (label entropy): {entropy:.4f}")
    print(f"Dirichlet alpha: {alpha}")
    print("-" * 50)

    # Test set (use last 20%)
    test_size = len(y) // 5
    X_test, y_test = X[-test_size:], y[-test_size:]
    X_train, y_train = X[:-test_size], y[:-test_size]

    # Aggregators to compare
    aggregators = {
        "FedAvg": FedAvg(),
        "FedSim": FedSim(similarity_threshold=0.5),
        "pFedSim": PFedSim(similarity_threshold=0.5, personalization_weight=0.3),
    }

    results = {name: [] for name in aggregators}

    for name, aggregator in aggregators.items():
        print(f"\n=== {name} ===")
        global_model = MLP(input_dim=X.shape[1], hidden_dim=64, num_classes=num_classes)

        for round_idx in range(num_rounds):
            new_state = train_round(
                global_model,
                client_indices,
                X_train,
                y_train,
                aggregator,
            )
            global_model.load_state_dict(new_state)

            acc = evaluate(global_model, X_test, y_test)
            results[name].append(acc)
            print(f"Round {round_idx + 1}: Accuracy = {acc:.4f}")

    # Summary
    print("\n" + "=" * 50)
    print("Final Accuracies:")
    for name, accs in results.items():
        print(f"  {name}: {accs[-1]:.4f}")


if __name__ == "__main__":
    main()
