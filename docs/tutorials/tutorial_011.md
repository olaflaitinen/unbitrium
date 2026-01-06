# Tutorial 011: FedAvg Algorithm Implementation

---

## Metadata

| Property | Value |
|----------|-------|
| **Tutorial ID** | 011 |
| **Title** | FedAvg Algorithm Implementation |
| **Category** | Core Algorithms |
| **Difficulty** | Intermediate |
| **Duration** | 75 minutes |
| **Prerequisites** | Tutorial 001-010 |
| **Author** | Unbitrium Contributors |
| **Last Updated** | January 2026 |

---

## Learning Objectives

By the end of this tutorial, you will be able to:

1. **Understand** the complete FedAvg algorithm including client selection, local training, and weighted aggregation.

2. **Implement** a production-ready FedAvg system with proper error handling and logging.

3. **Configure** hyperparameters including local epochs, batch size, learning rate, and client fraction.

4. **Analyze** convergence behavior under different configurations and data distributions.

5. **Debug** common issues in FedAvg implementations including weight divergence and slow convergence.

6. **Extend** the base FedAvg implementation with advanced features like learning rate scheduling.

---

## Prerequisites

Before starting this tutorial, ensure you have:

- **Completed Tutorials**: 001-010 (Fundamentals)
- **Knowledge**: Neural network training, gradient descent
- **Libraries**: PyTorch, NumPy
- **Hardware**: CPU sufficient, GPU optional

```python
# Verify prerequisites
import torch
import torch.nn as nn
import numpy as np

print(f"PyTorch: {torch.__version__}")
print(f"NumPy: {np.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
```

---

## Background and Theory

### The FedAvg Algorithm

**Federated Averaging (FedAvg)** is the foundational algorithm for federated learning, introduced by McMahan et al. (2017).

The algorithm optimizes:
$$\min_{\theta} F(\theta) = \sum_{k=1}^{K} \frac{n_k}{n} F_k(\theta)$$

where $F_k(\theta) = \mathbb{E}_{(x,y) \sim D_k}[\ell(\theta; x, y)]$

### Algorithm Steps

```
Algorithm: FedAvg
Input: K clients, T rounds, E local epochs, B batch size, η learning rate

1. Server initializes θ₀
2. for t = 1 to T do
3.     Select subset S_t of clients (fraction C)
4.     for each client k ∈ S_t in parallel do
5.         θ_k^{t+1} ← ClientUpdate(k, θ_t)
6.     end for
7.     θ_{t+1} ← Σ_{k∈S_t} (n_k/Σ_{j∈S_t} n_j) θ_k^{t+1}
8. end for
```

### Key Hyperparameters

| Parameter | Symbol | Typical Range | Impact |
|-----------|--------|---------------|--------|
| **Client fraction** | C | 0.1 - 1.0 | Communication efficiency |
| **Local epochs** | E | 1 - 20 | Computation vs accuracy |
| **Batch size** | B | 10 - 128 | Gradient variance |
| **Learning rate** | η | 0.001 - 0.1 | Convergence speed |

### Convergence Analysis

For FedAvg with non-convex objectives:

$$\mathbb{E}[\|\nabla F(\bar{\theta}^T)\|^2] \leq O\left(\frac{1}{\sqrt{KET}} + \frac{E}{T}\right)$$

where K is clients per round, E is local epochs, T is rounds.

---

## Architecture Diagram

```mermaid
flowchart TB
    subgraph "Server"
        INIT[Initialize θ₀]
        SELECT[Select Clients]
        DISTRIBUTE[Distribute θ_t]
        AGGREGATE[Weighted Average]
        UPDATE[Update θ_{t+1}]
    end

    subgraph "Client k"
        RECEIVE[Receive θ_t]
        LOCAL[E epochs SGD]
        SEND[Send θ_k^{t+1}]
    end

    INIT --> SELECT --> DISTRIBUTE
    DISTRIBUTE --> RECEIVE --> LOCAL --> SEND
    SEND --> AGGREGATE --> UPDATE
    UPDATE --> SELECT
```

---

## Implementation Code

### Part 1: Core Data Structures

```python
#!/usr/bin/env python3
"""
Tutorial 011: FedAvg Algorithm Implementation

This tutorial provides a complete, production-ready implementation
of the Federated Averaging algorithm.

Author: Unbitrium Contributors
License: EUPL-1.2
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class FedAvgConfig:
    """Configuration for FedAvg algorithm."""
    num_rounds: int = 100
    num_clients: int = 100
    client_fraction: float = 0.1  # C in paper
    local_epochs: int = 5  # E in paper
    batch_size: int = 32  # B in paper
    learning_rate: float = 0.01  # η in paper
    lr_decay: float = 0.995
    min_lr: float = 0.001
    momentum: float = 0.0
    weight_decay: float = 0.0
    seed: int = 42


@dataclass
class ClientState:
    """State maintained for each client."""
    client_id: int
    num_samples: int
    model_state: dict = field(default_factory=dict)
    metrics: dict = field(default_factory=dict)


@dataclass
class RoundMetrics:
    """Metrics collected per round."""
    round_num: int
    num_clients: int
    avg_loss: float
    accuracy: float
    learning_rate: float
    client_metrics: list = field(default_factory=list)


class SimpleDataset(Dataset):
    """Simple dataset for FL experiments."""

    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray,
    ) -> None:
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.labels[idx]
```

### Part 2: Model Definition

```python
class FederatedModel(nn.Module):
    """Neural network model for federated learning."""

    def __init__(
        self,
        input_dim: int = 32,
        hidden_dim: int = 64,
        num_classes: int = 10,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def create_model(
    input_dim: int = 32,
    num_classes: int = 10,
    model_type: str = "mlp",
) -> nn.Module:
    """Factory function to create models."""
    if model_type == "mlp":
        return FederatedModel(input_dim=input_dim, num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
```

### Part 3: FedAvg Client

```python
class FedAvgClient:
    """Federated learning client implementing local training."""

    def __init__(
        self,
        client_id: int,
        dataset: Dataset,
        config: FedAvgConfig,
        device: torch.device = None,
    ) -> None:
        self.client_id = client_id
        self.dataset = dataset
        self.config = config
        self.device = device or torch.device("cpu")
        
        self.dataloader = DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=True,
        )

    @property
    def num_samples(self) -> int:
        return len(self.dataset)

    def train(
        self,
        model: nn.Module,
        current_round: int = 0,
    ) -> dict[str, Any]:
        """Perform local training.

        Args:
            model: Model with global weights.
            current_round: Current round number.

        Returns:
            Dictionary with updated state_dict and metrics.
        """
        model = copy.deepcopy(model).to(self.device)
        model.train()

        # Compute effective learning rate with decay
        lr = max(
            self.config.min_lr,
            self.config.learning_rate * (self.config.lr_decay ** current_round),
        )

        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=self.config.momentum,
            weight_decay=self.config.weight_decay,
        )

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for epoch in range(self.config.local_epochs):
            epoch_loss = 0.0
            
            for batch_idx, (features, labels) in enumerate(self.dataloader):
                features = features.to(self.device)
                labels = labels.to(self.device)

                optimizer.zero_grad()
                outputs = model(features)
                loss = F.cross_entropy(outputs, labels)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                
                # Track accuracy
                preds = outputs.argmax(dim=1)
                total_correct += (preds == labels).sum().item()
                total_samples += labels.size(0)

            total_loss += epoch_loss

        # Compute metrics
        avg_loss = total_loss / (self.config.local_epochs * len(self.dataloader))
        accuracy = total_correct / total_samples if total_samples > 0 else 0.0

        return {
            "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
            "num_samples": self.num_samples,
            "client_id": self.client_id,
            "loss": avg_loss,
            "accuracy": accuracy,
            "learning_rate": lr,
        }
```

### Part 4: FedAvg Server

```python
class FedAvgServer:
    """Federated learning server implementing FedAvg."""

    def __init__(
        self,
        model: nn.Module,
        clients: list[FedAvgClient],
        config: FedAvgConfig,
        device: torch.device = None,
    ) -> None:
        self.model = model
        self.clients = clients
        self.config = config
        self.device = device or torch.device("cpu")
        
        self.model.to(self.device)
        self.history: list[RoundMetrics] = []

        # Set random seed
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)

    def select_clients(self) -> list[FedAvgClient]:
        """Select subset of clients for current round."""
        num_selected = max(1, int(len(self.clients) * self.config.client_fraction))
        indices = np.random.choice(
            len(self.clients),
            size=num_selected,
            replace=False,
        )
        return [self.clients[i] for i in indices]

    def aggregate(
        self,
        client_updates: list[dict[str, Any]],
    ) -> None:
        """Aggregate client updates using weighted averaging."""
        total_samples = sum(u["num_samples"] for u in client_updates)
        
        if total_samples == 0:
            logger.warning("No samples in client updates")
            return

        # Initialize aggregated state
        global_state = self.model.state_dict()
        new_state = {}

        for key in global_state.keys():
            new_state[key] = torch.zeros_like(global_state[key], dtype=torch.float32)
            
            for update in client_updates:
                weight = update["num_samples"] / total_samples
                new_state[key] += weight * update["state_dict"][key].float()

        self.model.load_state_dict(new_state)

    def train_round(self, round_num: int) -> RoundMetrics:
        """Execute one round of federated training."""
        # Select clients
        selected_clients = self.select_clients()
        logger.debug(f"Round {round_num}: selected {len(selected_clients)} clients")

        # Distribute global model and collect updates
        client_updates = []
        
        for client in selected_clients:
            # Client receives current global model
            update = client.train(self.model, round_num)
            client_updates.append(update)

        # Aggregate updates
        self.aggregate(client_updates)

        # Compute round metrics
        avg_loss = np.mean([u["loss"] for u in client_updates])
        avg_accuracy = np.mean([u["accuracy"] for u in client_updates])
        avg_lr = np.mean([u["learning_rate"] for u in client_updates])

        metrics = RoundMetrics(
            round_num=round_num,
            num_clients=len(selected_clients),
            avg_loss=avg_loss,
            accuracy=avg_accuracy,
            learning_rate=avg_lr,
            client_metrics=[{
                "client_id": u["client_id"],
                "loss": u["loss"],
                "accuracy": u["accuracy"],
            } for u in client_updates],
        )

        self.history.append(metrics)
        return metrics

    def train(self) -> list[RoundMetrics]:
        """Run full federated training."""
        logger.info(f"Starting FedAvg training: {self.config.num_rounds} rounds")
        logger.info(f"Clients: {len(self.clients)}, Fraction: {self.config.client_fraction}")

        for round_num in range(self.config.num_rounds):
            metrics = self.train_round(round_num)

            if (round_num + 1) % 10 == 0:
                logger.info(
                    f"Round {round_num + 1}/{self.config.num_rounds}: "
                    f"loss={metrics.avg_loss:.4f}, acc={metrics.accuracy:.4f}"
                )

        logger.info(f"Training complete. Final accuracy: {self.history[-1].accuracy:.4f}")
        return self.history

    def evaluate(self, test_data: Dataset) -> dict[str, float]:
        """Evaluate global model on test data."""
        self.model.eval()
        loader = DataLoader(test_data, batch_size=128)
        
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for features, labels in loader:
                features = features.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(features)
                loss = F.cross_entropy(outputs, labels)
                
                total_loss += loss.item() * labels.size(0)
                preds = outputs.argmax(dim=1)
                total_correct += (preds == labels).sum().item()
                total_samples += labels.size(0)

        return {
            "loss": total_loss / total_samples,
            "accuracy": total_correct / total_samples,
        }
```

### Part 5: Complete Example

```python
def generate_federated_data(
    num_clients: int = 100,
    samples_per_client: int = 100,
    num_classes: int = 10,
    feature_dim: int = 32,
    alpha: float = 0.5,
) -> list[Dataset]:
    """Generate non-IID data for clients using Dirichlet."""
    np.random.seed(42)
    
    # Dirichlet distribution for label proportions
    label_distributions = np.random.dirichlet(
        [alpha] * num_classes,
        num_clients,
    )
    
    datasets = []
    
    for client_id in range(num_clients):
        labels = np.random.choice(
            num_classes,
            size=samples_per_client,
            p=label_distributions[client_id],
        )
        
        features = np.random.randn(samples_per_client, feature_dim).astype(np.float32)
        for i, label in enumerate(labels):
            features[i, label % feature_dim] += 2.0
            features[i, (label * 3) % feature_dim] += 1.5
        
        datasets.append(SimpleDataset(features, labels))
    
    return datasets


def run_fedavg_experiment(
    alpha: float = 0.5,
    num_rounds: int = 50,
) -> dict[str, Any]:
    """Run complete FedAvg experiment."""
    config = FedAvgConfig(
        num_rounds=num_rounds,
        num_clients=100,
        client_fraction=0.1,
        local_epochs=5,
    )
    
    # Generate data
    client_datasets = generate_federated_data(
        num_clients=config.num_clients,
        alpha=alpha,
    )
    
    # Create model
    model = create_model(input_dim=32, num_classes=10)
    
    # Create clients
    clients = [
        FedAvgClient(i, dataset, config)
        for i, dataset in enumerate(client_datasets)
    ]
    
    # Create server and train
    server = FedAvgServer(model, clients, config)
    history = server.train()
    
    return {
        "config": config,
        "history": history,
        "final_accuracy": history[-1].accuracy,
    }


if __name__ == "__main__":
    results = run_fedavg_experiment(alpha=0.5, num_rounds=50)
    print(f"Final accuracy: {results['final_accuracy']:.4f}")
```

---

## Metrics and Evaluation

### Convergence Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| **Training Loss** | Average client loss | < 0.5 |
| **Accuracy** | Global model accuracy | > 80% |
| **Rounds to Converge** | Rounds to 90% of final | < 50 |

### Hyperparameter Impact

| Parameter | Too Low | Too High |
|-----------|---------|----------|
| **E (epochs)** | Slow convergence | Client drift |
| **C (fraction)** | High variance | Slow rounds |
| **η (LR)** | Slow convergence | Oscillation |

---

## Exercises

### Exercise 1: Learning Rate Scheduling

**Task**: Implement cosine annealing learning rate schedule.

### Exercise 2: Gradient Clipping

**Task**: Add gradient clipping to prevent exploding gradients.

### Exercise 3: Client Weighting

**Task**: Experiment with different client weighting schemes.

### Exercise 4: Momentum

**Task**: Add server-side momentum to aggregation.

---

## References

1. McMahan, B., et al. (2017). Communication-efficient learning of deep networks from decentralized data. In *AISTATS*.

2. Li, T., et al. (2020). Federated optimization in heterogeneous networks. In *MLSys*.

3. Reddi, S. J., et al. (2021). Adaptive federated optimization. In *ICLR*.

4. Wang, J., et al. (2021). A field guide to federated optimization. *arXiv*.

5. Karimireddy, S. P., et al. (2020). SCAFFOLD: Stochastic controlled averaging. In *ICML*.

---

*Copyright 2026 Olaf Yunus Laitinen Imanov and Contributors. Released under EUPL 1.2.*
