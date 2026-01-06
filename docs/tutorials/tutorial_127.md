# Tutorial 127: FL Robustness

---

## Metadata

| Property | Value |
|----------|-------|
| **Tutorial ID** | 127 |
| **Title** | Federated Learning Robustness |
| **Category** | Security |
| **Difficulty** | Advanced |
| **Duration** | 120 minutes |
| **Prerequisites** | Tutorial 001-126 |
| **Author** | Unbitrium Contributors |
| **Last Updated** | January 2026 |

---

## Learning Objectives

By completing this tutorial, you will be able to:

1. **Understand** robustness challenges in FL
2. **Implement** Byzantine-robust aggregation
3. **Design** attack-resistant FL systems
4. **Analyze** defense mechanisms
5. **Deploy** robust FL pipelines

---

## Prerequisites

Before starting this tutorial, ensure you have:

- Completed tutorials 001-126
- Understanding of FL fundamentals
- Knowledge of adversarial ML
- Familiarity with robust statistics

---

## Background and Theory

### Threat Model

FL faces various attacks:
- Byzantine clients: arbitrary malicious behavior
- Data poisoning: corrupt local data
- Model poisoning: corrupt model updates
- Free-riding: fake contributions

### Robust Aggregation Methods

```
Robust Aggregation:
├── Coordinate-wise
│   ├── Median
│   ├── Trimmed mean
│   └── Coordinate-wise clipping
├── Geometric
│   ├── Geometric median
│   ├── Krum
│   └── Multi-Krum
├── Statistical
│   ├── Robust covariance
│   ├── Spectral methods
│   └── HDBSCAN clustering
└── Learning-based
    ├── Attention aggregation
    ├── Trust scores
    └── Anomaly detection
```

---

## Implementation Code

```python
#!/usr/bin/env python3
"""
Tutorial 127: Federated Learning Robustness

This module implements Byzantine-robust FL with
various defense mechanisms against malicious clients.

Copyright 2026 Olaf Yunus Laitinen Imanov and Contributors
Released under EUPL 1.2
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
import copy
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AttackType(Enum):
    """Types of attacks."""
    NONE = "none"
    RANDOM = "random"
    SIGN_FLIP = "sign_flip"
    SCALE = "scale"
    LABEL_FLIP = "label_flip"


class DefenseType(Enum):
    """Types of defenses."""
    NONE = "none"
    MEDIAN = "median"
    TRIMMED_MEAN = "trimmed_mean"
    KRUM = "krum"
    MULTI_KRUM = "multi_krum"


@dataclass
class RobustConfig:
    """Configuration for robust FL."""

    num_rounds: int = 50
    num_clients: int = 20
    clients_per_round: int = 15

    input_dim: int = 32
    hidden_dim: int = 64
    num_classes: int = 10

    learning_rate: float = 0.01
    batch_size: int = 32
    local_epochs: int = 3

    # Attack parameters
    byzantine_ratio: float = 0.2
    attack_type: AttackType = AttackType.SIGN_FLIP

    # Defense parameters
    defense_type: DefenseType = DefenseType.KRUM
    trim_ratio: float = 0.2

    seed: int = 42


class RobustDataset(Dataset):
    """Dataset for robustness experiments."""

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


class RobustModel(nn.Module):
    """Model for robustness experiments."""

    def __init__(self, config: RobustConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RobustAggregator:
    """Robust aggregation methods."""

    def __init__(self, config: RobustConfig):
        self.config = config

    def aggregate(
        self,
        updates: List[Dict],
        defense: DefenseType
    ) -> Dict[str, torch.Tensor]:
        """Aggregate using specified defense."""
        if defense == DefenseType.NONE:
            return self._fedavg(updates)
        elif defense == DefenseType.MEDIAN:
            return self._median(updates)
        elif defense == DefenseType.TRIMMED_MEAN:
            return self._trimmed_mean(updates)
        elif defense == DefenseType.KRUM:
            return self._krum(updates, multi=False)
        elif defense == DefenseType.MULTI_KRUM:
            return self._krum(updates, multi=True)

        return self._fedavg(updates)

    def _fedavg(self, updates: List[Dict]) -> Dict[str, torch.Tensor]:
        """Standard FedAvg."""
        total_samples = sum(u["num_samples"] for u in updates)
        new_state = {}

        for key in updates[0]["state_dict"]:
            new_state[key] = sum(
                (u["num_samples"] / total_samples) * u["state_dict"][key].float()
                for u in updates
            )

        return new_state

    def _median(self, updates: List[Dict]) -> Dict[str, torch.Tensor]:
        """Coordinate-wise median."""
        new_state = {}

        for key in updates[0]["state_dict"]:
            stacked = torch.stack([u["state_dict"][key] for u in updates])
            new_state[key] = torch.median(stacked, dim=0).values

        return new_state

    def _trimmed_mean(self, updates: List[Dict]) -> Dict[str, torch.Tensor]:
        """Trimmed mean aggregation."""
        new_state = {}
        k = int(len(updates) * self.config.trim_ratio)

        for key in updates[0]["state_dict"]:
            stacked = torch.stack([u["state_dict"][key] for u in updates])
            sorted_vals, _ = torch.sort(stacked, dim=0)

            if k > 0 and 2*k < len(updates):
                trimmed = sorted_vals[k:-k]
            else:
                trimmed = sorted_vals

            new_state[key] = trimmed.mean(dim=0)

        return new_state

    def _krum(
        self,
        updates: List[Dict],
        multi: bool = False
    ) -> Dict[str, torch.Tensor]:
        """Krum or Multi-Krum aggregation."""
        n = len(updates)
        f = int(n * self.config.byzantine_ratio)

        # Flatten updates
        flattened = []
        for u in updates:
            flat = torch.cat([v.flatten() for v in u["state_dict"].values()])
            flattened.append(flat)

        # Compute pairwise distances
        distances = torch.zeros(n, n)
        for i in range(n):
            for j in range(i + 1, n):
                dist = torch.norm(flattened[i] - flattened[j])
                distances[i, j] = dist
                distances[j, i] = dist

        # Compute scores (sum of closest n-f-2 distances)
        scores = []
        for i in range(n):
            sorted_dists = torch.sort(distances[i]).values
            score = sorted_dists[:n-f-1].sum()
            scores.append(score.item())

        if multi:
            # Multi-Krum: average of n-f best
            sorted_indices = np.argsort(scores)
            selected = sorted_indices[:n-f]
            selected_updates = [updates[i] for i in selected]
            return self._fedavg(selected_updates)
        else:
            # Krum: single best
            best_idx = np.argmin(scores)
            return updates[best_idx]["state_dict"]


class ByzantineClient:
    """Client with possible Byzantine behavior."""

    def __init__(
        self,
        client_id: int,
        dataset: RobustDataset,
        config: RobustConfig,
        is_byzantine: bool = False
    ):
        self.client_id = client_id
        self.dataset = dataset
        self.config = config
        self.is_byzantine = is_byzantine

    def train(self, model: nn.Module) -> Dict[str, Any]:
        """Train or attack."""
        if self.is_byzantine:
            return self._attack(model)

        return self._honest_train(model)

    def _honest_train(self, model: nn.Module) -> Dict[str, Any]:
        """Honest training."""
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

        return {
            "state_dict": {k: v.cpu() for k, v in local.state_dict().items()},
            "num_samples": len(self.dataset),
            "avg_loss": total_loss / num_batches,
            "client_id": self.client_id,
            "is_byzantine": False
        }

    def _attack(self, model: nn.Module) -> Dict[str, Any]:
        """Generate attack update."""
        attack = self.config.attack_type

        # Get honest update first
        honest = self._honest_train(model)
        state_dict = honest["state_dict"]

        if attack == AttackType.RANDOM:
            # Random noise
            for key in state_dict:
                state_dict[key] = torch.randn_like(state_dict[key])

        elif attack == AttackType.SIGN_FLIP:
            # Flip signs and scale up
            for key in state_dict:
                state_dict[key] = -state_dict[key] * 5

        elif attack == AttackType.SCALE:
            # Scale up updates
            for key in state_dict:
                state_dict[key] = state_dict[key] * 100

        honest["state_dict"] = state_dict
        honest["is_byzantine"] = True

        return honest


class RobustServer:
    """Server with robust aggregation."""

    def __init__(
        self,
        model: nn.Module,
        clients: List[ByzantineClient],
        test_data: RobustDataset,
        config: RobustConfig
    ):
        self.model = model
        self.clients = clients
        self.test_data = test_data
        self.config = config

        self.aggregator = RobustAggregator(config)
        self.history: List[Dict] = []

    def evaluate(self) -> Dict[str, float]:
        """Evaluate model."""
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
        """Run robust FL training."""
        logger.info(
            f"Starting robust FL: {self.config.defense_type.value} "
            f"vs {self.config.attack_type.value}"
        )

        for round_num in range(self.config.num_rounds):
            # Select clients
            n = min(self.config.clients_per_round, len(self.clients))
            indices = np.random.choice(len(self.clients), n, replace=False)
            selected = [self.clients[i] for i in indices]

            # Collect updates
            updates = [c.train(self.model) for c in selected]

            # Robust aggregation
            new_state = self.aggregator.aggregate(
                updates,
                self.config.defense_type
            )
            self.model.load_state_dict(new_state)

            # Count Byzantine
            num_byzantine = sum(1 for u in updates if u.get("is_byzantine", False))

            # Evaluate
            metrics = self.evaluate()

            record = {
                "round": round_num,
                **metrics,
                "num_byzantine": num_byzantine,
                "num_clients": len(updates)
            }
            self.history.append(record)

            if (round_num + 1) % 10 == 0:
                logger.info(
                    f"Round {round_num + 1}: acc={metrics['accuracy']:.4f}, "
                    f"byzantine={num_byzantine}/{len(updates)}"
                )

        return self.history


def main():
    """Main entry point."""
    print("=" * 60)
    print("Tutorial 127: FL Robustness")
    print("=" * 60)

    config = RobustConfig()
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    # Create clients (some Byzantine)
    clients = []
    num_byzantine = int(config.num_clients * config.byzantine_ratio)

    for i in range(config.num_clients):
        is_byzantine = i < num_byzantine
        dataset = RobustDataset(client_id=i, dim=config.input_dim, seed=config.seed)
        client = ByzantineClient(i, dataset, config, is_byzantine)
        clients.append(client)

    np.random.shuffle(clients)

    test_data = RobustDataset(client_id=999, n=300, seed=999)

    # Compare defenses
    results = {}
    for defense in DefenseType:
        config.defense_type = defense
        model = RobustModel(config)
        server = RobustServer(model, clients, test_data, config)
        history = server.train()
        results[defense.value] = history[-1]["accuracy"]

    print("\n" + "=" * 60)
    print("Defense Comparison")
    for defense, acc in results.items():
        print(f"  {defense}: {acc:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
```

---

## Key Insights

### Robustness Best Practices

1. **Know your threat**: Design defense for expected attacks
2. **Combine methods**: Multi-layer defense
3. **Validate clients**: Trust but verify
4. **Monitor metrics**: Detect anomalies

---

## Exercises

1. **Exercise 1**: Implement FoolsGold
2. **Exercise 2**: Add adaptive attack
3. **Exercise 3**: Design anomaly detection
4. **Exercise 4**: Combine multiple defenses

---

## References

1. Blanchard, P., et al. (2017). Machine learning with adversaries: Byzantine tolerant gradient descent. In *NeurIPS*.
2. Fung, C., et al. (2020). FoolsGold: Mitigating sybils in FL. In *MLSys*.
3. Yin, D., et al. (2018). Byzantine-robust distributed learning. In *ICML*.

---

*Copyright 2026 Olaf Yunus Laitinen Imanov and Contributors. Released under EUPL 1.2.*
