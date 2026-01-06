# Tutorial 119: FL Client Selection

---

## Metadata

| Property | Value |
|----------|-------|
| **Tutorial ID** | 119 |
| **Title** | Federated Learning Client Selection |
| **Category** | Systems |
| **Difficulty** | Advanced |
| **Duration** | 90 minutes |
| **Prerequisites** | Tutorial 001-118 |
| **Author** | Unbitrium Contributors |
| **Last Updated** | January 2026 |

---

## Learning Objectives

By completing this tutorial, you will be able to:

1. **Understand** client selection strategies
2. **Implement** intelligent selection algorithms
3. **Design** fair and efficient selection
4. **Analyze** selection impact on training
5. **Deploy** adaptive client selection

---

## Prerequisites

Before starting this tutorial, ensure you have:

- Completed tutorials 001-118
- Understanding of FL fundamentals
- Knowledge of sampling strategies
- Familiarity with system constraints

---

## Background and Theory

### Client Selection Strategies

```
Selection Strategies:
├── Random Selection
│   ├── Uniform random
│   ├── Stratified sampling
│   └── Cluster-based
├── Performance-Based
│   ├── Data quality
│   ├── Model contribution
│   └── Update quality
├── Resource-Based
│   ├── Availability
│   ├── Bandwidth
│   └── Compute capacity
└── Fairness-Aware
    ├── Participation quotas
    ├── Round-robin
    └── Age-based
```

---

## Implementation Code

```python
#!/usr/bin/env python3
"""
Tutorial 119: FL Client Selection

This module implements various client selection
strategies for federated learning.

Copyright 2026 Olaf Yunus Laitinen Imanov and Contributors
Released under EUPL 1.2
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from enum import Enum
import copy
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SelectionStrategy(Enum):
    RANDOM = "random"
    POWER_OF_CHOICE = "power_of_choice"
    CONTRIBUTION = "contribution"
    FAIR = "fair"
    RESOURCE = "resource"


@dataclass
class SelectionConfig:
    """Selection configuration."""

    num_rounds: int = 50
    num_clients: int = 30
    clients_per_round: int = 10

    input_dim: int = 32
    hidden_dim: int = 64
    num_classes: int = 10

    learning_rate: float = 0.01
    batch_size: int = 32
    local_epochs: int = 3

    strategy: SelectionStrategy = SelectionStrategy.FAIR

    seed: int = 42


class ClientSelector:
    """Client selection algorithms."""

    def __init__(self, num_clients: int, seed: int = 0):
        self.num_clients = num_clients
        self.rng = np.random.RandomState(seed)

        self.participation_counts = {i: 0 for i in range(num_clients)}
        self.contribution_scores = {i: 1.0 for i in range(num_clients)}
        self.last_selected = {i: -1 for i in range(num_clients)}

    def select(
        self,
        k: int,
        strategy: SelectionStrategy,
        round_num: int,
        client_info: Optional[Dict[int, Dict]] = None
    ) -> List[int]:
        """Select k clients."""
        if strategy == SelectionStrategy.RANDOM:
            return self._random_select(k)
        elif strategy == SelectionStrategy.POWER_OF_CHOICE:
            return self._power_of_choice(k, client_info)
        elif strategy == SelectionStrategy.CONTRIBUTION:
            return self._contribution_select(k)
        elif strategy == SelectionStrategy.FAIR:
            return self._fair_select(k, round_num)
        elif strategy == SelectionStrategy.RESOURCE:
            return self._resource_select(k, client_info)

        return self._random_select(k)

    def _random_select(self, k: int) -> List[int]:
        """Uniform random selection."""
        return list(self.rng.choice(self.num_clients, k, replace=False))

    def _power_of_choice(
        self,
        k: int,
        client_info: Optional[Dict] = None
    ) -> List[int]:
        """Power-of-d-choices selection."""
        d = min(2 * k, self.num_clients)
        candidates = self.rng.choice(self.num_clients, d, replace=False)

        # Select k with least participation
        scored = [(c, self.participation_counts[c]) for c in candidates]
        scored.sort(key=lambda x: x[1])

        return [c for c, _ in scored[:k]]

    def _contribution_select(self, k: int) -> List[int]:
        """Select based on contribution scores."""
        scores = np.array([self.contribution_scores[i] for i in range(self.num_clients)])
        probs = scores / scores.sum()

        selected = []
        remaining = list(range(self.num_clients))

        for _ in range(k):
            probs_norm = probs[remaining] / probs[remaining].sum()
            idx = self.rng.choice(len(remaining), p=probs_norm)
            selected.append(remaining[idx])
            remaining.pop(idx)

        return selected

    def _fair_select(self, k: int, round_num: int) -> List[int]:
        """Fair selection prioritizing least participation."""
        eligible = list(range(self.num_clients))

        # Score by staleness and participation
        scores = []
        for c in eligible:
            staleness = round_num - self.last_selected[c]
            participation = self.participation_counts[c]

            score = staleness * 10 - participation
            scores.append((c, score))

        scores.sort(key=lambda x: x[1], reverse=True)

        return [c for c, _ in scores[:k]]

    def _resource_select(
        self,
        k: int,
        client_info: Optional[Dict] = None
    ) -> List[int]:
        """Select based on resources."""
        if not client_info:
            return self._random_select(k)

        # Score by resource availability
        scored = []
        for c in range(self.num_clients):
            info = client_info.get(c, {})
            availability = info.get("availability", 1.0)
            bandwidth = info.get("bandwidth", 1.0)

            score = availability * bandwidth
            scored.append((c, score))

        scored.sort(key=lambda x: x[1], reverse=True)

        return [c for c, _ in scored[:k]]

    def update_stats(
        self,
        selected: List[int],
        round_num: int,
        contributions: Optional[Dict[int, float]] = None
    ) -> None:
        """Update selection statistics."""
        for c in selected:
            self.participation_counts[c] += 1
            self.last_selected[c] = round_num

            if contributions and c in contributions:
                # EMA of contribution
                self.contribution_scores[c] = (
                    0.9 * self.contribution_scores[c] +
                    0.1 * contributions[c]
                )


class SelectDataset(Dataset):
    def __init__(self, n: int = 200, dim: int = 32, classes: int = 10, seed: int = 0):
        np.random.seed(seed)
        self.x = torch.randn(n, dim, dtype=torch.float32)
        self.y = torch.randint(0, classes, (n,), dtype=torch.long)
        for i in range(n):
            self.x[i, self.y[i].item() % dim] += 2.0

    def __len__(self): return len(self.y)
    def __getitem__(self, idx): return self.x[idx], self.y[idx]


class SelectModel(nn.Module):
    def __init__(self, config: SelectionConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.num_classes)
        )

    def forward(self, x): return self.net(x)


class SelectClient:
    def __init__(
        self,
        client_id: int,
        dataset: SelectDataset,
        config: SelectionConfig
    ):
        self.client_id = client_id
        self.dataset = dataset
        self.config = config

        # Simulated resources
        self.availability = np.random.uniform(0.5, 1.0)
        self.bandwidth = np.random.uniform(0.3, 1.0)

    def get_info(self) -> Dict:
        return {
            "availability": self.availability,
            "bandwidth": self.bandwidth,
            "data_size": len(self.dataset)
        }

    def train(self, model: nn.Module) -> Dict:
        local = copy.deepcopy(model)
        optimizer = torch.optim.SGD(local.parameters(), lr=self.config.learning_rate)
        loader = DataLoader(self.dataset, batch_size=self.config.batch_size, shuffle=True)

        local.train()
        total_loss, num_batches = 0.0, 0
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
            "avg_loss": total_loss / num_batches,
            "client_id": self.client_id
        }


class SelectServer:
    def __init__(
        self,
        model: nn.Module,
        clients: List[SelectClient],
        test_data: SelectDataset,
        config: SelectionConfig
    ):
        self.model = model
        self.clients = clients
        self.test_data = test_data
        self.config = config

        self.selector = ClientSelector(len(clients), config.seed)
        self.history: List[Dict] = []

    def aggregate(self, updates: List[Dict]) -> None:
        total = sum(u["num_samples"] for u in updates)
        new_state = {}
        for key in updates[0]["state_dict"]:
            new_state[key] = sum(
                (u["num_samples"] / total) * u["state_dict"][key].float()
                for u in updates
            )
        self.model.load_state_dict(new_state)

    def evaluate(self) -> Dict[str, float]:
        self.model.eval()
        loader = DataLoader(self.test_data, batch_size=64)
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in loader:
                pred = self.model(x).argmax(dim=1)
                correct += (pred == y).sum().item()
                total += len(y)
        return {"accuracy": correct / total}

    def train(self) -> List[Dict]:
        logger.info(f"Starting FL with {self.config.strategy.value} selection")

        client_info = {c.client_id: c.get_info() for c in self.clients}

        for round_num in range(self.config.num_rounds):
            selected_ids = self.selector.select(
                self.config.clients_per_round,
                self.config.strategy,
                round_num,
                client_info
            )

            selected = [self.clients[i] for i in selected_ids]
            updates = [c.train(self.model) for c in selected]

            self.aggregate(updates)

            # Update selector
            contributions = {u["client_id"]: 1.0 / (u["avg_loss"] + 0.1) for u in updates}
            self.selector.update_stats(selected_ids, round_num, contributions)

            metrics = self.evaluate()

            # Participation stats
            counts = self.selector.participation_counts
            fairness = np.std(list(counts.values()))

            record = {
                "round": round_num,
                **metrics,
                "fairness_std": fairness
            }
            self.history.append(record)

            if (round_num + 1) % 10 == 0:
                logger.info(f"Round {round_num + 1}: acc={metrics['accuracy']:.4f}")

        return self.history


def main():
    print("=" * 60)
    print("Tutorial 119: FL Client Selection")
    print("=" * 60)

    config = SelectionConfig()
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    clients = [
        SelectClient(i, SelectDataset(seed=config.seed + i), config)
        for i in range(config.num_clients)
    ]
    test_data = SelectDataset(seed=999)

    # Compare strategies
    results = {}
    for strategy in SelectionStrategy:
        config.strategy = strategy
        model = SelectModel(config)
        server = SelectServer(model, clients, test_data, config)

        # Reset selector
        server.selector = ClientSelector(len(clients), config.seed)

        history = server.train()
        results[strategy.value] = {
            "accuracy": history[-1]["accuracy"],
            "fairness": history[-1]["fairness_std"]
        }

    print("\n" + "=" * 60)
    print("Selection Strategy Comparison")
    for strategy, r in results.items():
        print(f"  {strategy}: acc={r['accuracy']:.4f}, fairness={r['fairness']:.2f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
```

---

## Key Insights

### Selection Best Practices

1. **Balance fairness**: All clients should participate
2. **Consider resources**: Match workload to capacity
3. **Track contribution**: Reward helpful clients
4. **Adapt dynamically**: Change strategy as needed

---

## Exercises

1. **Exercise 1**: Add multi-armed bandit selection
2. **Exercise 2**: Implement active learning selection
3. **Exercise 3**: Design cluster-based selection
4. **Exercise 4**: Add deadline-aware selection

---

## References

1. Cho, Y.J., et al. (2020). Client selection for FL. In *ICML*.
2. Nishio, T., & Yonetani, R. (2019). Client selection for FL with heterogeneous resources. In *ICC*.
3. Li, T., et al. (2020). Fair resource allocation in FL. In *ICLR*.

---

*Copyright 2026 Olaf Yunus Laitinen Imanov and Contributors. Released under EUPL 1.2.*
