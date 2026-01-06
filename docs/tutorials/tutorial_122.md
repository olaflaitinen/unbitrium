# Tutorial 122: FL Neural Architecture Search

---

## Metadata

| Property | Value |
|----------|-------|
| **Tutorial ID** | 122 |
| **Title** | Federated Neural Architecture Search |
| **Category** | Automation |
| **Difficulty** | Expert |
| **Duration** | 120 minutes |
| **Prerequisites** | Tutorial 001-121 |
| **Author** | Unbitrium Contributors |
| **Last Updated** | January 2026 |

---

## Learning Objectives

By completing this tutorial, you will be able to:

1. **Understand** NAS challenges in FL
2. **Implement** federated architecture search
3. **Design** distributed search strategies
4. **Analyze** architecture performance
5. **Deploy** auto-designed FL models

---

## Prerequisites

Before starting this tutorial, ensure you have:

- Completed tutorials 001-121
- Understanding of FL fundamentals
- Knowledge of neural architecture search
- Familiarity with search spaces

---

## Background and Theory

### FL NAS Challenges

NAS in FL faces unique difficulties:
- Architecture evaluation is expensive
- Distributed data affects optimal architecture
- Client heterogeneity needs consideration
- Communication overhead from search

### Search Space Design

```
FL NAS Search Space:
├── Layer Types
│   ├── Linear layers
│   ├── Convolutions
│   ├── Skip connections
│   └── Attention
├── Layer Parameters
│   ├── Hidden dimensions
│   ├── Kernel sizes
│   └── Activation functions
└── Architecture Patterns
    ├── Depth
    ├── Width
    └── Connectivity
```

---

## Implementation Code

```python
#!/usr/bin/env python3
"""
Tutorial 122: Federated Neural Architecture Search

This module implements NAS for finding optimal
architectures in federated learning settings.

Copyright 2026 Olaf Yunus Laitinen Imanov and Contributors
Released under EUPL 1.2
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
import copy
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LayerType(Enum):
    """Available layer types."""
    LINEAR = "linear"
    CONV1D = "conv1d"
    SKIP = "skip"


@dataclass
class NASConfig:
    """Configuration for FL NAS."""

    num_architecture_samples: int = 20
    num_rounds_per_arch: int = 10

    num_clients: int = 10
    clients_per_round: int = 5

    input_dim: int = 32
    num_classes: int = 10

    # Search space
    min_layers: int = 1
    max_layers: int = 4
    hidden_dims: List[int] = field(default_factory=lambda: [32, 64, 128])

    learning_rate: float = 0.01
    batch_size: int = 32
    local_epochs: int = 2

    seed: int = 42


@dataclass
class Architecture:
    """Neural network architecture specification."""

    layers: List[Dict[str, Any]]

    def __str__(self) -> str:
        desc = []
        for i, layer in enumerate(self.layers):
            desc.append(f"L{i}: {layer['type']} ({layer.get('dim', 'N/A')})")
        return " -> ".join(desc)


class ArchitectureSearchSpace:
    """Search space for architectures."""

    def __init__(self, config: NASConfig, seed: int = 0):
        self.config = config
        self.rng = np.random.RandomState(seed)

    def sample(self) -> Architecture:
        """Sample random architecture."""
        num_layers = self.rng.randint(
            self.config.min_layers,
            self.config.max_layers + 1
        )

        layers = []
        for i in range(num_layers):
            layer_type = self.rng.choice([LayerType.LINEAR, LayerType.LINEAR])
            hidden_dim = self.rng.choice(self.config.hidden_dims)
            dropout = self.rng.uniform(0, 0.5)

            layers.append({
                "type": layer_type.value,
                "dim": hidden_dim,
                "dropout": dropout
            })

        return Architecture(layers=layers)

    def mutate(self, arch: Architecture) -> Architecture:
        """Mutate architecture."""
        new_layers = copy.deepcopy(arch.layers)

        mutation = self.rng.choice(["add", "remove", "modify"])

        if mutation == "add" and len(new_layers) < self.config.max_layers:
            idx = self.rng.randint(0, len(new_layers) + 1)
            new_layer = {
                "type": LayerType.LINEAR.value,
                "dim": self.rng.choice(self.config.hidden_dims),
                "dropout": self.rng.uniform(0, 0.3)
            }
            new_layers.insert(idx, new_layer)

        elif mutation == "remove" and len(new_layers) > self.config.min_layers:
            idx = self.rng.randint(0, len(new_layers))
            new_layers.pop(idx)

        elif mutation == "modify" and len(new_layers) > 0:
            idx = self.rng.randint(0, len(new_layers))
            new_layers[idx]["dim"] = self.rng.choice(self.config.hidden_dims)

        return Architecture(layers=new_layers)


def build_model(arch: Architecture, config: NASConfig) -> nn.Module:
    """Build model from architecture."""
    layers = []
    in_dim = config.input_dim

    for layer_spec in arch.layers:
        out_dim = layer_spec["dim"]
        layers.append(nn.Linear(in_dim, out_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(layer_spec["dropout"]))
        in_dim = out_dim

    layers.append(nn.Linear(in_dim, config.num_classes))

    return nn.Sequential(*layers)


class NASDataset(Dataset):
    """Dataset for NAS experiments."""

    def __init__(
        self,
        client_id: int,
        n: int = 200,
        dim: int = 32,
        classes: int = 10,
        seed: int = 0
    ):
        np.random.seed(seed + client_id)

        self.x = torch.randn(n, dim, dtype=torch.float32)
        self.y = torch.randint(0, classes, (n,), dtype=torch.long)

        for i in range(n):
            self.x[i, self.y[i].item() % dim] += 2.0

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx]


class NASClient:
    """Client for NAS experiments."""

    def __init__(
        self,
        client_id: int,
        dataset: NASDataset,
        config: NASConfig
    ):
        self.client_id = client_id
        self.dataset = dataset
        self.config = config

    def train(self, model: nn.Module) -> Dict[str, Any]:
        """Train model."""
        local = copy.deepcopy(model)
        optimizer = torch.optim.Adam(
            local.parameters(),
            lr=self.config.learning_rate
        )

        loader = DataLoader(
            self.dataset,
            batch_size=self.config.batch_size,
            shuffle=True
        )

        local.train()
        total_loss = 0.0
        num_batches = 0

        for _ in range(self.config.local_epochs):
            for x, y in loader:
                optimizer.zero_grad()
                loss = F.cross_entropy(local(x), y)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                num_batches += 1

        return {
            "state_dict": {k: v.cpu() for k, v in local.state_dict().items()},
            "num_samples": len(self.dataset),
            "avg_loss": total_loss / num_batches
        }


class NASServer:
    """Server for federated NAS."""

    def __init__(
        self,
        clients: List[NASClient],
        test_data: NASDataset,
        config: NASConfig
    ):
        self.clients = clients
        self.test_data = test_data
        self.config = config

        self.search_space = ArchitectureSearchSpace(config, config.seed)

        self.arch_history: List[Dict] = []
        self.best_arch: Optional[Architecture] = None
        self.best_score: float = 0.0

    def evaluate(self, model: nn.Module) -> float:
        """Evaluate model."""
        model.eval()
        loader = DataLoader(self.test_data, batch_size=64)

        correct, total = 0, 0
        with torch.no_grad():
            for x, y in loader:
                pred = model(x).argmax(dim=1)
                correct += (pred == y).sum().item()
                total += len(y)

        return correct / total

    def evaluate_architecture(self, arch: Architecture) -> float:
        """Train and evaluate architecture."""
        model = build_model(arch, self.config)

        for _ in range(self.config.num_rounds_per_arch):
            n = min(self.config.clients_per_round, len(self.clients))
            indices = np.random.choice(len(self.clients), n, replace=False)
            selected = [self.clients[i] for i in indices]

            updates = [c.train(model) for c in selected]

            # Aggregate
            total_samples = sum(u["num_samples"] for u in updates)
            new_state = {}

            for key in updates[0]["state_dict"]:
                new_state[key] = sum(
                    (u["num_samples"] / total_samples) * u["state_dict"][key].float()
                    for u in updates
                )

            model.load_state_dict(new_state)

        return self.evaluate(model)

    def search(self) -> Architecture:
        """Run NAS."""
        logger.info(f"Starting FL NAS with {self.config.num_architecture_samples} samples")

        for i in range(self.config.num_architecture_samples):
            # Sample or mutate
            if self.best_arch and np.random.random() > 0.3:
                arch = self.search_space.mutate(self.best_arch)
            else:
                arch = self.search_space.sample()

            # Evaluate
            score = self.evaluate_architecture(arch)
            params = sum(
                np.prod(layer["dim"] for layer in arch.layers if "dim" in layer)
                for _ in [1]  # Rough estimate
            )

            self.arch_history.append({
                "index": i,
                "architecture": str(arch),
                "score": score,
                "num_layers": len(arch.layers)
            })

            if score > self.best_score:
                self.best_score = score
                self.best_arch = arch
                logger.info(f"New best architecture at sample {i}: {score:.4f}")
                logger.info(f"  Architecture: {arch}")

        return self.best_arch


def main():
    """Main entry point."""
    print("=" * 60)
    print("Tutorial 122: FL NAS")
    print("=" * 60)

    config = NASConfig()
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    # Create clients
    clients = []
    for i in range(config.num_clients):
        dataset = NASDataset(client_id=i, dim=config.input_dim, seed=config.seed)
        client = NASClient(i, dataset, config)
        clients.append(client)

    test_data = NASDataset(client_id=999, n=300, seed=999)

    # Run NAS
    server = NASServer(clients, test_data, config)
    best_arch = server.search()

    print("\n" + "=" * 60)
    print("NAS Complete")
    print(f"Best Score: {server.best_score:.4f}")
    print(f"Best Architecture: {best_arch}")
    print(f"Number of layers: {len(best_arch.layers)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
```

---

## Key Insights

### FL NAS Strategies

1. **Weight sharing**: Reduce evaluation cost
2. **Progressive search**: Start simple
3. **Client-aware**: Consider heterogeneity
4. **Early stopping**: Quick rejection

---

## Exercises

1. **Exercise 1**: Add DARTS-style differentiable NAS
2. **Exercise 2**: Implement weight sharing
3. **Exercise 3**: Design cell search space
4. **Exercise 4**: Add multi-objective NAS

---

## References

1. Zoph, B., & Le, Q.V. (2017). Neural architecture search with RL. In *ICLR*.
2. Liu, H., et al. (2019). DARTS: Differentiable architecture search. In *ICLR*.
3. Xu, J., et al. (2021). Federated neural architecture search. In *NeurIPS Workshop*.

---

*Copyright 2026 Olaf Yunus Laitinen Imanov and Contributors. Released under EUPL 1.2.*
