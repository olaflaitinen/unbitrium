# Tutorial 087: FL with Momentum

---

## Metadata

| Property | Value |
|----------|-------|
| **Tutorial ID** | 087 |
| **Title** | FL with Momentum |
| **Category** | Optimization |
| **Difficulty** | Intermediate |
| **Duration** | 90 minutes |
| **Prerequisites** | Tutorial 001-086 |
| **Author** | Unbitrium Contributors |
| **Last Updated** | January 2026 |

---

## Learning Objectives

By the end of this tutorial, you will be able to:

1. **Understand** momentum in federated optimization.
2. **Implement** FedAvgM with server-side momentum.
3. **Design** client and server momentum strategies.
4. **Analyze** convergence improvements from momentum.
5. **Apply** adaptive momentum techniques.
6. **Evaluate** momentum impact on non-IID data.
7. **Compare** various momentum configurations.

---

## Prerequisites

- **Completed Tutorials**: 001-086
- **Knowledge**: Optimization, momentum methods
- **Libraries**: PyTorch, NumPy

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from torch.utils.data import Dataset, DataLoader
import copy

print(f"PyTorch: {torch.__version__}")
```

---

## Background and Theory

### Momentum in Optimization

| Method | Description | Update Rule |
|--------|-------------|-------------|
| Standard SGD | No momentum | θ = θ - η∇L |
| Heavy Ball | Classical momentum | v = βv + ∇L; θ = θ - ηv |
| Nesterov | Look-ahead gradient | v = βv + ∇L(θ - βv); θ = θ - ηv |
| Adam | Adaptive moment | m,v updated; θ = θ - η·m/√v |

### Momentum in FL

| Position | Description | Benefits |
|----------|-------------|----------|
| Client-side | Each client uses momentum | Faster local training |
| Server-side | Server applies momentum | Smooths aggregated updates |
| Both | Combined approach | Best convergence |

### FedAvgM Algorithm

```mermaid
graph TB
    subgraph "Server"
        GLOBAL[Global Model w_t]
        PSEUDO[Pseudo-gradient Δ_t]
        MOMENTUM[Momentum Buffer m_t]
        UPDATE[Update w_{t+1}]
    end

    subgraph "Clients"
        LOCAL[Local Training]
        SEND[Send Updates]
    end

    GLOBAL --> LOCAL --> SEND --> PSEUDO
    MOMENTUM --> UPDATE
    PSEUDO --> UPDATE
    UPDATE --> MOMENTUM
    UPDATE --> GLOBAL
```

### Mathematical Formulation

Server momentum update:
$$m_{t+1} = \beta m_t + \Delta_t$$
$$w_{t+1} = w_t - \eta_s m_{t+1}$$

Where:
- $\Delta_t$ is the aggregated pseudo-gradient
- $\beta$ is the momentum coefficient
- $\eta_s$ is the server learning rate

---

## Implementation Code

### Part 1: Configuration and Components

```python
#!/usr/bin/env python3
"""
Tutorial 087: FL with Momentum

Comprehensive implementation of momentum-based federated learning
including server-side, client-side, and combined momentum strategies.

Author: Unbitrium Contributors
License: EUPL-1.2
"""

from __future__ import annotations
import copy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


class MomentumType(Enum):
    """Types of momentum placement."""
    NONE = "none"
    SERVER = "server"
    CLIENT = "client"
    BOTH = "both"
    NESTEROV_SERVER = "nesterov_server"


@dataclass
class MomentumConfig:
    """Configuration for momentum FL."""

    # General
    num_rounds: int = 100
    num_clients: int = 20
    clients_per_round: int = 10
    local_epochs: int = 5
    batch_size: int = 32
    seed: int = 42

    # Learning rates
    client_lr: float = 0.01
    server_lr: float = 1.0

    # Momentum
    momentum_type: MomentumType = MomentumType.SERVER
    server_momentum: float = 0.9
    client_momentum: float = 0.9
    nesterov: bool = False

    # Model
    input_dim: int = 32
    hidden_dim: int = 128
    num_classes: int = 10

    # Data heterogeneity
    alpha: float = 0.5  # Dirichlet parameter


class MomentumDataset(Dataset):
    """Non-IID dataset for momentum experiments."""

    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        client_id: int = 0,
    ):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
        self.client_id = client_id

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.labels[idx]


class MomentumModel(nn.Module):
    """Model for momentum experiments."""

    def __init__(self, config: MomentumConfig):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(config.hidden_dim // 2, config.num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def create_non_iid_data(
    config: MomentumConfig,
) -> Tuple[List[MomentumDataset], MomentumDataset]:
    """Create non-IID data using Dirichlet distribution."""
    np.random.seed(config.seed)

    # Generate base data
    n_train = config.num_clients * 100
    n_test = 500

    train_features = np.random.randn(n_train, config.input_dim).astype(np.float32)
    train_labels = np.random.randint(0, config.num_classes, n_train)

    # Add class-specific patterns
    for i in range(n_train):
        train_features[i, train_labels[i] % config.input_dim] += 2.0

    # Dirichlet partitioning
    label_distributions = np.random.dirichlet(
        [config.alpha] * config.num_classes,
        config.num_clients
    )

    # Organize by class
    class_indices = {c: np.where(train_labels == c)[0] for c in range(config.num_classes)}

    client_datasets = []
    for client_id in range(config.num_clients):
        client_indices = []

        for c in range(config.num_classes):
            n_class = int(label_distributions[client_id, c] * 100)
            if n_class > 0 and len(class_indices[c]) > 0:
                selected = np.random.choice(
                    class_indices[c],
                    min(n_class, len(class_indices[c])),
                    replace=True
                )
                client_indices.extend(selected.tolist())

        if len(client_indices) > 0:
            client_datasets.append(MomentumDataset(
                train_features[client_indices],
                train_labels[client_indices],
                client_id
            ))

    # Test data
    test_features = np.random.randn(n_test, config.input_dim).astype(np.float32)
    test_labels = np.random.randint(0, config.num_classes, n_test)
    for i in range(n_test):
        test_features[i, test_labels[i] % config.input_dim] += 2.0

    test_dataset = MomentumDataset(test_features, test_labels, -1)

    return client_datasets, test_dataset
```

### Part 2: Client with Momentum

```python
class MomentumClient:
    """FL client with optional client-side momentum."""

    def __init__(
        self,
        client_id: int,
        dataset: MomentumDataset,
        config: MomentumConfig,
    ):
        self.client_id = client_id
        self.dataset = dataset
        self.config = config

        # Client-side momentum buffer (persists across rounds)
        self.momentum_buffer: Dict[str, torch.Tensor] = {}

    @property
    def num_samples(self) -> int:
        return len(self.dataset)

    def _use_client_momentum(self) -> bool:
        return self.config.momentum_type in [
            MomentumType.CLIENT,
            MomentumType.BOTH
        ]

    def train(self, model: nn.Module) -> Dict[str, Any]:
        """Train locally with optional momentum."""
        local_model = copy.deepcopy(model)

        # Choose optimizer based on momentum setting
        if self._use_client_momentum():
            optimizer = torch.optim.SGD(
                local_model.parameters(),
                lr=self.config.client_lr,
                momentum=self.config.client_momentum,
                nesterov=self.config.nesterov,
            )
        else:
            optimizer = torch.optim.SGD(
                local_model.parameters(),
                lr=self.config.client_lr,
            )

        loader = DataLoader(
            self.dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
        )

        local_model.train()
        total_loss = 0
        num_batches = 0

        for _ in range(self.config.local_epochs):
            for features, labels in loader:
                optimizer.zero_grad()
                outputs = local_model(features)
                loss = F.cross_entropy(outputs, labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                num_batches += 1

        return {
            "state_dict": {k: v.cpu() for k, v in local_model.state_dict().items()},
            "num_samples": self.num_samples,
            "loss": total_loss / num_batches if num_batches > 0 else 0,
        }


class MomentumServer:
    """FL server with server-side momentum (FedAvgM)."""

    def __init__(
        self,
        model: nn.Module,
        clients: List[MomentumClient],
        test_dataset: MomentumDataset,
        config: MomentumConfig,
    ):
        self.model = model
        self.clients = clients
        self.test_dataset = test_dataset
        self.config = config
        self.history: List[Dict] = []

        # Server momentum buffer
        self.momentum_buffer: Dict[str, torch.Tensor] = {}
        self._initialize_momentum_buffer()

    def _initialize_momentum_buffer(self) -> None:
        """Initialize momentum buffer to zeros."""
        for name, param in self.model.named_parameters():
            self.momentum_buffer[name] = torch.zeros_like(param)

    def _use_server_momentum(self) -> bool:
        return self.config.momentum_type in [
            MomentumType.SERVER,
            MomentumType.BOTH,
            MomentumType.NESTEROV_SERVER,
        ]

    def _use_nesterov(self) -> bool:
        return self.config.momentum_type == MomentumType.NESTEROV_SERVER

    def aggregate_with_momentum(self, updates: List[Dict]) -> None:
        """Aggregate updates with server-side momentum."""
        total_samples = sum(u["num_samples"] for u in updates)

        # Compute pseudo-gradient (difference from global model)
        pseudo_gradient = {}
        for name, param in self.model.named_parameters():
            # Weighted average of client models
            aggregated = sum(
                (u["num_samples"] / total_samples) * u["state_dict"][name].float()
                for u in updates
            )
            # Pseudo-gradient is current - aggregated (pointing towards aggregated)
            pseudo_gradient[name] = param.data - aggregated

        # Apply momentum
        if self._use_server_momentum():
            for name, param in self.model.named_parameters():
                # Update momentum buffer
                self.momentum_buffer[name] = (
                    self.config.server_momentum * self.momentum_buffer[name] +
                    pseudo_gradient[name]
                )

                # Nesterov look-ahead
                if self._use_nesterov():
                    look_ahead = (
                        self.config.server_momentum * self.momentum_buffer[name] +
                        pseudo_gradient[name]
                    )
                    param.data -= self.config.server_lr * look_ahead
                else:
                    param.data -= self.config.server_lr * self.momentum_buffer[name]
        else:
            # No momentum, standard FedAvg
            for name, param in self.model.named_parameters():
                aggregated = sum(
                    (u["num_samples"] / total_samples) * u["state_dict"][name].float()
                    for u in updates
                )
                param.data.copy_(aggregated)

    def evaluate(self) -> Tuple[float, float]:
        """Evaluate global model."""
        self.model.eval()
        loader = DataLoader(self.test_dataset, batch_size=128)

        correct, total, total_loss = 0, 0, 0.0
        with torch.no_grad():
            for features, labels in loader:
                outputs = self.model(features)
                loss = F.cross_entropy(outputs, labels)
                preds = outputs.argmax(dim=1)

                correct += (preds == labels).sum().item()
                total += len(labels)
                total_loss += loss.item() * len(labels)

        return correct / total, total_loss / total

    def train(self) -> List[Dict]:
        """Run FL training with momentum."""
        for round_num in range(self.config.num_rounds):
            # Select clients
            selected = np.random.choice(
                self.clients,
                min(self.config.clients_per_round, len(self.clients)),
                replace=False,
            )

            # Collect updates
            updates = [c.train(self.model) for c in selected]

            # Aggregate with momentum
            self.aggregate_with_momentum(updates)

            # Evaluate
            acc, loss = self.evaluate()
            avg_client_loss = np.mean([u["loss"] for u in updates])

            self.history.append({
                "round": round_num,
                "accuracy": acc,
                "loss": loss,
                "client_loss": avg_client_loss,
            })

            if (round_num + 1) % 10 == 0:
                print(f"Round {round_num + 1}: acc={acc:.4f}, loss={loss:.4f}")

        return self.history


def compare_momentum_strategies():
    """Compare different momentum strategies."""
    results = {}

    for momentum_type in [MomentumType.NONE, MomentumType.SERVER, MomentumType.BOTH]:
        config = MomentumConfig(
            num_rounds=50,
            num_clients=10,
            momentum_type=momentum_type,
        )

        torch.manual_seed(config.seed)
        np.random.seed(config.seed)

        client_datasets, test_dataset = create_non_iid_data(config)
        clients = [
            MomentumClient(i, dataset, config)
            for i, dataset in enumerate(client_datasets)
        ]

        model = MomentumModel(config)
        server = MomentumServer(model, clients, test_dataset, config)

        print(f"\n=== {momentum_type.value} ===")
        history = server.train()

        results[momentum_type.value] = {
            "final_accuracy": history[-1]["accuracy"],
            "best_accuracy": max(h["accuracy"] for h in history),
        }

    print("\n=== Summary ===")
    for name, metrics in results.items():
        print(f"{name}: final={metrics['final_accuracy']:.4f}, "
              f"best={metrics['best_accuracy']:.4f}")


if __name__ == "__main__":
    compare_momentum_strategies()
```

---

## Exercises

1. **Exercise 1**: Implement adaptive server momentum.
2. **Exercise 2**: Compare with Adam-style adaptive moment.
3. **Exercise 3**: Analyze momentum buffer growth.
4. **Exercise 4**: Add warm-up for momentum.
5. **Exercise 5**: Implement per-layer momentum rates.

---

## References

1. Hsu, T. H., et al. (2019). Measuring non-identical data distribution effects. *arXiv*.
2. Reddi, S., et al. (2021). Adaptive federated optimization. In *ICLR*.
3. Wang, J., et al. (2020). SlowMo: Improving communication-efficient DL. In *ICLR*.
4. Yu, H., et al. (2019). Parallel restarted SGD with faster convergence. In *AAAI*.
5. Karimireddy, S. P., et al. (2020). SCAFFOLD: Stochastic controlled averaging. In *ICML*.

---

*Copyright 2026 Olaf Yunus Laitinen Imanov and Contributors. Released under EUPL 1.2.*
