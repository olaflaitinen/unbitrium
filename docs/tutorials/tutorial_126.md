# Tutorial 126: FL Fairness

---

## Metadata

| Property | Value |
|----------|-------|
| **Tutorial ID** | 126 |
| **Title** | Federated Learning Fairness |
| **Category** | Ethics |
| **Difficulty** | Advanced |
| **Duration** | 120 minutes |
| **Prerequisites** | Tutorial 001-125 |
| **Author** | Unbitrium Contributors |
| **Last Updated** | January 2026 |

---

## Learning Objectives

By completing this tutorial, you will be able to:

1. **Understand** fairness challenges in FL
2. **Implement** fair federated learning algorithms
3. **Measure** fairness metrics across clients
4. **Design** equitable FL systems
5. **Balance** accuracy and fairness tradeoffs

---

## Prerequisites

Before starting this tutorial, ensure you have:

- Completed tutorials 001-125
- Understanding of FL fundamentals
- Knowledge of ML fairness concepts
- Familiarity with optimization

---

## Background and Theory

### Fairness in FL

FL introduces unique fairness challenges:
- Performance varies across clients
- Data heterogeneity causes disparities
- Underrepresented clients perform worse
- Majority clients dominate aggregation

### Fairness Types

```
FL Fairness Taxonomy:
├── Client-Level Fairness
│   ├── Performance parity
│   ├── Participation fairness
│   └── Resource fairness
├── Group Fairness
│   ├── Demographic parity
│   ├── Equal opportunity
│   └── Equalized odds
├── Individual Fairness
│   ├── Similar treatment
│   └── Consistency
└── Distributive Fairness
    ├── Rawlsian (maximin)
    └── Utilitarian
```

### Fairness Metrics

| Metric | Definition | Target |
|--------|------------|--------|
| Accuracy Parity | max - min accuracy | Low |
| Variance | std(accuracies) | Low |
| Worst-case | min(accuracies) | High |
| Jain Index | Σxᵢ² / (n * Σxᵢ)² | Close to 1 |

---

## Implementation Code

```python
#!/usr/bin/env python3
"""
Tutorial 126: Federated Learning Fairness

This module implements fair FL algorithms including
AFL, q-FedAvg, and fairness metrics.

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
import copy
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class FairnessConfig:
    """Configuration for fair FL."""

    num_rounds: int = 50
    num_clients: int = 20
    clients_per_round: int = 10

    input_dim: int = 32
    hidden_dim: int = 64
    num_classes: int = 10

    learning_rate: float = 0.01
    batch_size: int = 32
    local_epochs: int = 3

    # Fairness parameters
    q_param: float = 0.5   # q-FedAvg fairness parameter
    lambda_fair: float = 0.1  # Fairness regularization

    seed: int = 42


class FairnessDataset(Dataset):
    """Dataset with controllable heterogeneity."""

    def __init__(
        self,
        client_id: int,
        n: int = 200,
        dim: int = 32,
        classes: int = 10,
        seed: int = 0,
        difficulty: float = 1.0  # Higher = harder
    ):
        np.random.seed(seed + client_id)

        self.difficulty = difficulty

        # Add noise proportional to difficulty
        self.x = torch.randn(n, dim, dtype=torch.float32) * difficulty
        self.y = torch.randint(0, classes, (n,), dtype=torch.long)

        # Weaker signal for higher difficulty
        for i in range(n):
            self.x[i, self.y[i].item() % dim] += 2.0 / difficulty

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx]


class FairnessModel(nn.Module):
    """Standard model for fairness experiments."""

    def __init__(self, config: FairnessConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FairnessMetrics:
    """Fairness metrics calculator."""

    @staticmethod
    def compute_all(accuracies: List[float]) -> Dict[str, float]:
        """Compute all fairness metrics."""
        arr = np.array(accuracies)

        metrics = {
            "mean_accuracy": np.mean(arr),
            "min_accuracy": np.min(arr),
            "max_accuracy": np.max(arr),
            "accuracy_gap": np.max(arr) - np.min(arr),
            "std_accuracy": np.std(arr),
            "jain_index": FairnessMetrics.jain_index(arr),
            "worst_10_pct": FairnessMetrics.worst_k_percent(arr, 10),
        }

        return metrics

    @staticmethod
    def jain_index(values: np.ndarray) -> float:
        """Jain's fairness index (1 = perfect equality)."""
        n = len(values)
        return (values.sum() ** 2) / (n * (values ** 2).sum() + 1e-8)

    @staticmethod
    def worst_k_percent(values: np.ndarray, k: int = 10) -> float:
        """Average of worst k percent."""
        n = max(1, int(len(values) * k / 100))
        return np.sort(values)[:n].mean()


class FairClient:
    """Client for fair FL experiments."""

    def __init__(
        self,
        client_id: int,
        dataset: FairnessDataset,
        config: FairnessConfig
    ):
        self.client_id = client_id
        self.dataset = dataset
        self.config = config
        self.last_loss = 1.0

    def train(self, model: nn.Module) -> Dict[str, Any]:
        local = copy.deepcopy(model)
        optimizer = torch.optim.SGD(
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

        self.last_loss = total_loss / num_batches

        return {
            "state_dict": {k: v.cpu() for k, v in local.state_dict().items()},
            "num_samples": len(self.dataset),
            "avg_loss": self.last_loss,
            "client_id": self.client_id
        }

    def evaluate(self, model: nn.Module) -> float:
        """Evaluate model on local data."""
        model.eval()
        loader = DataLoader(self.dataset, batch_size=64)

        correct, total = 0, 0
        with torch.no_grad():
            for x, y in loader:
                pred = model(x).argmax(dim=1)
                correct += (pred == y).sum().item()
                total += len(y)

        return correct / total


class FedAvgServer:
    """Standard FedAvg for comparison."""

    def __init__(
        self,
        model: nn.Module,
        clients: List[FairClient],
        config: FairnessConfig
    ):
        self.model = model
        self.clients = clients
        self.config = config

    def aggregate(self, updates: List[Dict]) -> None:
        total_samples = sum(u["num_samples"] for u in updates)
        new_state = {}

        for key in updates[0]["state_dict"]:
            new_state[key] = sum(
                (u["num_samples"] / total_samples) * u["state_dict"][key].float()
                for u in updates
            )

        self.model.load_state_dict(new_state)

    def evaluate_fairness(self) -> Dict[str, float]:
        """Evaluate fairness across all clients."""
        accuracies = [c.evaluate(self.model) for c in self.clients]
        return FairnessMetrics.compute_all(accuracies)


class QFedAvgServer:
    """
    q-FedAvg: Fair aggregation using reweighting.

    Gives higher weight to clients with higher loss,
    improving worst-case performance.
    """

    def __init__(
        self,
        model: nn.Module,
        clients: List[FairClient],
        config: FairnessConfig
    ):
        self.model = model
        self.clients = clients
        self.config = config
        self.delta_t: Dict[str, torch.Tensor] = {}

    def aggregate(self, updates: List[Dict]) -> None:
        """q-FedAvg aggregation."""
        q = self.config.q_param

        # Weight by loss raised to power q
        weights = []
        for u in updates:
            w = u["avg_loss"] ** q
            weights.append(w)

        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]

        new_state = {}
        for key in updates[0]["state_dict"]:
            new_state[key] = sum(
                w * u["state_dict"][key].float()
                for w, u in zip(weights, updates)
            )

        self.model.load_state_dict(new_state)

    def evaluate_fairness(self) -> Dict[str, float]:
        accuracies = [c.evaluate(self.model) for c in self.clients]
        return FairnessMetrics.compute_all(accuracies)


class AFLServer:
    """
    Agnostic Federated Learning.

    Optimizes for worst-case client performance.
    """

    def __init__(
        self,
        model: nn.Module,
        clients: List[FairClient],
        config: FairnessConfig
    ):
        self.model = model
        self.clients = clients
        self.config = config

        # Client weights (will adapt over time)
        self.client_weights = {c.client_id: 1.0 for c in clients}

    def update_weights(self, updates: List[Dict]) -> None:
        """Update client weights based on losses."""
        # Increase weights for high-loss clients
        losses = {u["client_id"]: u["avg_loss"] for u in updates}
        max_loss = max(losses.values())

        for client_id, loss in losses.items():
            # Multiplicative increase for high-loss clients
            ratio = loss / (max_loss + 1e-8)
            self.client_weights[client_id] *= (1 + ratio * 0.1)

        # Normalize
        total = sum(self.client_weights.values())
        for k in self.client_weights:
            self.client_weights[k] /= total

    def aggregate(self, updates: List[Dict]) -> None:
        """AFL aggregation with adaptive weights."""
        self.update_weights(updates)

        total_weight = sum(
            self.client_weights[u["client_id"]] for u in updates
        )

        new_state = {}
        for key in updates[0]["state_dict"]:
            new_state[key] = sum(
                (self.client_weights[u["client_id"]] / total_weight) *
                u["state_dict"][key].float()
                for u in updates
            )

        self.model.load_state_dict(new_state)

    def evaluate_fairness(self) -> Dict[str, float]:
        accuracies = [c.evaluate(self.model) for c in self.clients]
        return FairnessMetrics.compute_all(accuracies)


class PropFairServer:
    """
    Proportional Fairness.

    Maximizes geometric mean of client utilities.
    """

    def __init__(
        self,
        model: nn.Module,
        clients: List[FairClient],
        config: FairnessConfig
    ):
        self.model = model
        self.clients = clients
        self.config = config

        self.client_utilities: Dict[int, float] = {
            c.client_id: 0.5 for c in clients
        }

    def update_utilities(self, updates: List[Dict]) -> None:
        """Update client utilities."""
        for u in updates:
            client_id = u["client_id"]
            # Utility is 1 - loss (higher is better)
            new_util = max(0.01, 1.0 - u["avg_loss"])
            # Exponential moving average
            self.client_utilities[client_id] = (
                0.9 * self.client_utilities[client_id] + 0.1 * new_util
            )

    def aggregate(self, updates: List[Dict]) -> None:
        """Proportional fair aggregation."""
        self.update_utilities(updates)

        # Weight inversely proportional to utility
        weights = []
        for u in updates:
            util = self.client_utilities[u["client_id"]]
            w = 1.0 / (util + 0.1) * u["num_samples"]
            weights.append(w)

        total = sum(weights)
        weights = [w / total for w in weights]

        new_state = {}
        for key in updates[0]["state_dict"]:
            new_state[key] = sum(
                w * u["state_dict"][key].float()
                for w, u in zip(weights, updates)
            )

        self.model.load_state_dict(new_state)

    def evaluate_fairness(self) -> Dict[str, float]:
        accuracies = [c.evaluate(self.model) for c in self.clients]
        return FairnessMetrics.compute_all(accuracies)


def run_experiment(
    server_class,
    clients: List[FairClient],
    config: FairnessConfig,
    name: str
) -> Dict[str, float]:
    """Run fairness experiment."""
    model = FairnessModel(config)
    server = server_class(model, clients, config)

    for round_num in range(config.num_rounds):
        n = min(config.clients_per_round, len(clients))
        indices = np.random.choice(len(clients), n, replace=False)
        selected = [clients[i] for i in indices]

        updates = [c.train(server.model) for c in selected]
        server.aggregate(updates)

        if (round_num + 1) % 25 == 0:
            metrics = server.evaluate_fairness()
            logger.info(f"{name} Round {round_num + 1}: min={metrics['min_accuracy']:.4f}")

    return server.evaluate_fairness()


def main():
    """Main entry point."""
    print("=" * 60)
    print("Tutorial 126: FL Fairness")
    print("=" * 60)

    config = FairnessConfig()
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    # Create heterogeneous clients
    clients = []
    for i in range(config.num_clients):
        difficulty = 0.5 + (i / config.num_clients) * 2.0  # Varying difficulty
        dataset = FairnessDataset(
            client_id=i,
            dim=config.input_dim,
            seed=config.seed,
            difficulty=difficulty
        )
        client = FairClient(i, dataset, config)
        clients.append(client)

    # Run experiments
    results = {}

    for name, server_class in [
        ("FedAvg", FedAvgServer),
        ("q-FedAvg", QFedAvgServer),
        ("AFL", AFLServer),
        ("PropFair", PropFairServer),
    ]:
        # Reset clients
        for c in clients:
            c.last_loss = 1.0

        metrics = run_experiment(server_class, clients, config, name)
        results[name] = metrics

    # Print comparison
    print("\n" + "=" * 60)
    print("Fairness Comparison")
    print("=" * 60)

    for name, metrics in results.items():
        print(f"\n{name}:")
        print(f"  Mean Accuracy: {metrics['mean_accuracy']:.4f}")
        print(f"  Min Accuracy:  {metrics['min_accuracy']:.4f}")
        print(f"  Accuracy Gap:  {metrics['accuracy_gap']:.4f}")
        print(f"  Jain Index:    {metrics['jain_index']:.4f}")

    print("=" * 60)


if __name__ == "__main__":
    main()
```

---

## Key Insights

### Fairness-Accuracy Tradeoff

1. **FedAvg**: Best mean accuracy, worst min accuracy
2. **q-FedAvg**: Tunable tradeoff via q parameter
3. **AFL**: Best worst-case, may reduce mean
4. **PropFair**: Balanced approach

### Best Practices

- Measure multiple fairness metrics
- Consider client heterogeneity sources
- Use adaptive aggregation weights
- Monitor worst-case performance

---

## Exercises

1. **Exercise 1**: Implement group fairness constraints
2. **Exercise 2**: Add demographic parity
3. **Exercise 3**: Design fairness-aware client selection
4. **Exercise 4**: Visualize fairness over rounds

---

## References

1. Li, T., et al. (2020). Fair resource allocation in FL. In *ICLR*.
2. Mohri, M., et al. (2019). Agnostic federated learning. In *ICML*.
3. Li, Q., et al. (2021). Ditto: Fair and robust FL. In *ICML*.

---

*Copyright 2026 Olaf Yunus Laitinen Imanov and Contributors. Released under EUPL 1.2.*
