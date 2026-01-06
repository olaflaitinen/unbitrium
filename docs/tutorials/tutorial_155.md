# Tutorial 155: FL Quick Reference

---

## Metadata

| Property | Value |
|----------|-------|
| **Tutorial ID** | 155 |
| **Title** | Federated Learning Quick Reference |
| **Category** | Reference |
| **Difficulty** | All Levels |
| **Duration** | 60 minutes |
| **Prerequisites** | Basic Python, PyTorch |
| **Author** | Unbitrium Contributors |
| **Last Updated** | January 2026 |

---

## Learning Objectives

By completing this tutorial, you will be able to:

1. **Access** quick reference implementations
2. **Use** copy-paste code snippets
3. **Understand** FL algorithm patterns
4. **Apply** common FL utilities
5. **Reference** key formulas and concepts

---

## Prerequisites

Basic understanding of:
- Python programming
- PyTorch fundamentals
- Machine learning basics

---

## Quick Reference Overview

This tutorial serves as a comprehensive quick reference for federated
learning implementations. Use it as a lookup resource during development.

---

## Core Algorithm References

### FedAvg Algorithm

```python
def fedavg_aggregate(
    updates: List[Dict],
    weight_key: str = "num_samples"
) -> Dict[str, torch.Tensor]:
    """
    Federated Averaging aggregation.

    Args:
        updates: List of client updates with 'state_dict' and weight_key
        weight_key: Key for weighting (usually 'num_samples')

    Returns:
        Aggregated state dictionary

    Usage:
        updates = [{'state_dict': {...}, 'num_samples': 100}, ...]
        new_state = fedavg_aggregate(updates)
        model.load_state_dict(new_state)
    """
    total_weight = sum(u[weight_key] for u in updates)
    new_state = {}

    for key in updates[0]["state_dict"]:
        weighted_sum = sum(
            (u[weight_key] / total_weight) * u["state_dict"][key].float()
            for u in updates
        )
        new_state[key] = weighted_sum

    return new_state
```

### FedProx Algorithm

```python
def fedprox_loss(
    loss: torch.Tensor,
    local_model: nn.Module,
    global_model: nn.Module,
    mu: float = 0.01
) -> torch.Tensor:
    """
    FedProx loss with proximal term.

    Args:
        loss: Original task loss
        local_model: Local model being trained
        global_model: Global model (reference)
        mu: Proximal term coefficient

    Returns:
        Loss with proximal regularization

    Usage:
        loss = F.cross_entropy(output, target)
        loss = fedprox_loss(loss, local_model, global_model, mu=0.01)
        loss.backward()
    """
    proximal_term = 0.0
    for (name, local_param), (_, global_param) in zip(
        local_model.named_parameters(),
        global_model.named_parameters()
    ):
        proximal_term += ((local_param - global_param.detach()) ** 2).sum()

    return loss + (mu / 2) * proximal_term
```

### Gradient Clipping

```python
def clip_gradients(
    model: nn.Module,
    max_norm: float = 1.0
) -> float:
    """
    Clip gradients by norm.

    Args:
        model: Model with gradients
        max_norm: Maximum gradient norm

    Returns:
        Total gradient norm before clipping

    Usage:
        loss.backward()
        grad_norm = clip_gradients(model, max_norm=1.0)
        optimizer.step()
    """
    return torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_norm
    ).item()
```

---

## Complete Implementation Reference

```python
#!/usr/bin/env python3
"""
Tutorial 155: Federated Learning Quick Reference

Complete reference implementation with copy-paste snippets.

Copyright 2026 Olaf Yunus Laitinen Imanov and Contributors
Released under EUPL 1.2
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any, Callable
import copy


# =============================================================================
# Configuration Template
# =============================================================================

@dataclass
class FLConfig:
    """Standard FL configuration template."""

    # FL parameters
    num_rounds: int = 50
    num_clients: int = 10
    clients_per_round: int = 5

    # Model parameters
    input_dim: int = 32
    hidden_dim: int = 64
    num_classes: int = 10

    # Training parameters
    learning_rate: float = 0.01
    batch_size: int = 32
    local_epochs: int = 3

    # Advanced
    gradient_clip: float = 1.0
    weight_decay: float = 0.0
    momentum: float = 0.0

    # Reproducibility
    seed: int = 42


# =============================================================================
# Dataset Template
# =============================================================================

class FLDataset(Dataset):
    """Standard FL dataset template."""

    def __init__(
        self,
        n: int = 100,
        dim: int = 32,
        classes: int = 10,
        seed: int = 0
    ):
        np.random.seed(seed)
        self.x = torch.randn(n, dim, dtype=torch.float32)
        self.y = torch.randint(0, classes, (n,), dtype=torch.long)

        # Add class-specific patterns
        for i in range(n):
            self.x[i, self.y[i].item() % dim] += 2.0

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx]


# =============================================================================
# Model Templates
# =============================================================================

class SimpleMLP(nn.Module):
    """Simple MLP for classification."""

    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CNN(nn.Module):
    """Simple CNN for image classification."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        return self.fc(x)


# =============================================================================
# Client Template
# =============================================================================

class FLClient:
    """Standard FL client template."""

    def __init__(
        self,
        client_id: int,
        dataset: Dataset,
        config: FLConfig
    ):
        self.client_id = client_id
        self.dataset = dataset
        self.config = config

    def train(self, model: nn.Module) -> Dict[str, Any]:
        """Standard local training."""
        # Clone model
        local_model = copy.deepcopy(model)

        # Setup optimizer
        optimizer = torch.optim.SGD(
            local_model.parameters(),
            lr=self.config.learning_rate,
            momentum=self.config.momentum,
            weight_decay=self.config.weight_decay
        )

        # Data loader
        loader = DataLoader(
            self.dataset,
            batch_size=self.config.batch_size,
            shuffle=True
        )

        # Training loop
        local_model.train()
        total_loss = 0.0
        num_batches = 0

        for _ in range(self.config.local_epochs):
            for x, y in loader:
                optimizer.zero_grad()
                output = local_model(x)
                loss = F.cross_entropy(output, y)
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    local_model.parameters(),
                    self.config.gradient_clip
                )

                optimizer.step()
                total_loss += loss.item()
                num_batches += 1

        return {
            "state_dict": {
                k: v.cpu() for k, v in local_model.state_dict().items()
            },
            "num_samples": len(self.dataset),
            "avg_loss": total_loss / num_batches
        }


# =============================================================================
# Server Template
# =============================================================================

class FLServer:
    """Standard FL server template."""

    def __init__(
        self,
        model: nn.Module,
        clients: List[FLClient],
        test_data: Dataset,
        config: FLConfig
    ):
        self.model = model
        self.clients = clients
        self.test_data = test_data
        self.config = config

    def select_clients(self) -> List[FLClient]:
        """Random client selection."""
        n = min(self.config.clients_per_round, len(self.clients))
        indices = np.random.choice(len(self.clients), n, replace=False)
        return [self.clients[i] for i in indices]

    def aggregate(self, updates: List[Dict]) -> None:
        """FedAvg aggregation."""
        total_samples = sum(u["num_samples"] for u in updates)
        new_state = {}

        for key in updates[0]["state_dict"]:
            new_state[key] = sum(
                (u["num_samples"] / total_samples) * u["state_dict"][key].float()
                for u in updates
            )

        self.model.load_state_dict(new_state)

    def evaluate(self) -> Dict[str, float]:
        """Evaluate on test data."""
        self.model.eval()
        loader = DataLoader(self.test_data, batch_size=64)

        correct, total = 0, 0
        total_loss = 0.0

        with torch.no_grad():
            for x, y in loader:
                output = self.model(x)
                loss = F.cross_entropy(output, y)
                pred = output.argmax(dim=1)

                correct += (pred == y).sum().item()
                total += len(y)
                total_loss += loss.item() * len(y)

        return {
            "accuracy": correct / total,
            "loss": total_loss / total
        }

    def train(self) -> List[Dict]:
        """Run FL training."""
        history = []

        for round_num in range(self.config.num_rounds):
            # Select clients
            selected = self.select_clients()

            # Collect updates
            updates = [c.train(self.model) for c in selected]

            # Aggregate
            self.aggregate(updates)

            # Evaluate
            metrics = self.evaluate()

            record = {"round": round_num, **metrics}
            history.append(record)

            if (round_num + 1) % 10 == 0:
                print(f"Round {round_num + 1}: acc={metrics['accuracy']:.4f}")

        return history


# =============================================================================
# Utility Functions
# =============================================================================

def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_model_diff(
    model1: nn.Module,
    model2: nn.Module
) -> float:
    """Compute L2 difference between models."""
    diff = 0.0
    for (_, p1), (_, p2) in zip(
        model1.named_parameters(),
        model2.named_parameters()
    ):
        diff += ((p1 - p2) ** 2).sum().item()
    return diff ** 0.5


def create_non_iid_data(
    num_clients: int,
    samples_per_client: int,
    num_classes: int,
    alpha: float = 0.5
) -> List[FLDataset]:
    """Create non-IID data using Dirichlet distribution."""
    datasets = []

    for i in range(num_clients):
        # Dirichlet distribution for class imbalance
        class_probs = np.random.dirichlet([alpha] * num_classes)

        dataset = FLDataset(
            n=samples_per_client,
            classes=num_classes,
            seed=i
        )

        # Resample with class imbalance
        labels = np.random.choice(
            num_classes,
            size=samples_per_client,
            p=class_probs
        )
        dataset.y = torch.tensor(labels)

        datasets.append(dataset)

    return datasets


# =============================================================================
# Main Example
# =============================================================================

def main():
    """Quick start example."""
    print("=" * 60)
    print("Tutorial 155: FL Quick Reference")
    print("=" * 60)

    # Setup
    config = FLConfig()
    set_seed(config.seed)

    # Create components
    datasets = [FLDataset(seed=i) for i in range(config.num_clients)]
    clients = [FLClient(i, d, config) for i, d in enumerate(datasets)]
    test_data = FLDataset(n=200, seed=999)

    model = SimpleMLP(
        config.input_dim,
        config.hidden_dim,
        config.num_classes
    )
    print(f"Model parameters: {count_parameters(model):,}")

    # Train
    server = FLServer(model, clients, test_data, config)
    history = server.train()

    print(f"\nFinal Accuracy: {history[-1]['accuracy']:.4f}")


if __name__ == "__main__":
    main()
```

---

## Quick Reference Tables

### Aggregation Methods

| Method | Formula | When to Use |
|--------|---------|-------------|
| FedAvg | Σ(nᵢ/n)wᵢ | Standard FL |
| Uniform | (1/K)Σwᵢ | Equal clients |
| Median | median(wᵢ) | Byzantine robust |
| Trimmed Mean | trim(wᵢ) | Outlier robust |

### Learning Rate Guidelines

| Scenario | Recommended LR |
|----------|---------------|
| Standard | 0.01 |
| Non-IID | 0.001-0.01 |
| Many clients | 0.1 |
| Few clients | 0.001 |

### Local Epochs Guidelines

| Data Heterogeneity | Epochs |
|-------------------|--------|
| IID | 5-10 |
| Mild non-IID | 3-5 |
| Severe non-IID | 1-3 |

---

## Common Patterns

### Client Update Pattern

```python
update = {
    "state_dict": local_model.state_dict(),
    "num_samples": len(dataset),
    "metrics": {"loss": avg_loss}
}
```

### Training Loop Pattern

```python
for round in range(num_rounds):
    selected = select_clients()
    updates = [c.train(model) for c in selected]
    aggregate(updates)
    evaluate()
```

---

## References

1. McMahan, B., et al. (2017). Communication-efficient learning. *AISTATS*.
2. Li, T., et al. (2020). Federated optimization. *MLSys*.
3. Kairouz, P., et al. (2021). Advances in FL. *FnTML*.

---

*Copyright 2026 Olaf Yunus Laitinen Imanov and Contributors. Released under EUPL 1.2.*
