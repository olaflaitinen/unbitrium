# Tutorial 140: FL Data Valuation

---

## Metadata

| Property | Value |
|----------|-------|
| **Tutorial ID** | 140 |
| **Title** | Federated Learning Data Valuation |
| **Category** | Economics |
| **Difficulty** | Advanced |
| **Duration** | 120 minutes |
| **Prerequisites** | Tutorial 001-139 |
| **Author** | Unbitrium Contributors |
| **Last Updated** | January 2026 |

---

## Learning Objectives

By completing this tutorial, you will be able to:

1. **Understand** data valuation in FL
2. **Implement** Shapley-based valuation
3. **Design** contribution measurement systems
4. **Analyze** incentive mechanisms
5. **Deploy** fair reward distribution

---

## Prerequisites

Before starting this tutorial, ensure you have:

- Completed tutorials 001-139
- Understanding of FL fundamentals
- Knowledge of game theory basics
- Familiarity with incentive design

---

## Background and Theory

### Why Data Valuation?

In FL, clients contribute differently:
- Some have more data
- Some have higher quality data
- Some have more diverse data
- Contributions affect global model

Fair valuation enables:
- Incentive-compatible participation
- Fair reward distribution
- Quality-aware aggregation
- Sustainable FL ecosystems

### Valuation Methods

```
Data Valuation Methods:
├── Marginal Contribution
│   ├── Leave-one-out
│   └── Influence functions
├── Shapley Value
│   ├── Exact Shapley
│   ├── Monte Carlo Shapley
│   └── Federated Shapley
├── Data Quality
│   ├── Accuracy contribution
│   ├── Gradient quality
│   └── Data diversity
└── Market Mechanisms
    ├── Auction-based
    └── Posted-price
```

### Properties of Good Valuation

| Property | Description | Shapley |
|----------|-------------|---------|
| Efficiency | Total value distributed | ✅ |
| Symmetry | Equal contribution = equal value | ✅ |
| Null player | Zero contribution = zero value | ✅ |
| Additivity | Value of sum = sum of values | ✅ |

---

## Implementation Code

```python
#!/usr/bin/env python3
"""
Tutorial 140: Federated Learning Data Valuation

This module implements data valuation techniques including
Shapley value and contribution measurement.

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
from itertools import combinations
import copy
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ValuationConfig:
    """Configuration for data valuation."""

    num_rounds: int = 30
    num_clients: int = 10
    clients_per_round: int = 5

    input_dim: int = 32
    hidden_dim: int = 64
    num_classes: int = 10

    learning_rate: float = 0.01
    batch_size: int = 32
    local_epochs: int = 2

    # Valuation parameters
    shapley_samples: int = 100  # Monte Carlo samples

    seed: int = 42


class ValuationDataset(Dataset):
    """Dataset with controllable quality."""

    def __init__(
        self,
        client_id: int,
        n: int = 100,
        dim: int = 32,
        classes: int = 10,
        seed: int = 0,
        quality: float = 1.0,  # 0-1, higher is better
        noise_level: float = 0.0
    ):
        np.random.seed(seed + client_id)

        self.quality = quality
        self.x = torch.randn(n, dim, dtype=torch.float32)
        self.y = torch.randint(0, classes, (n,), dtype=torch.long)

        # Add signal
        signal_strength = 2.0 * quality
        for i in range(n):
            self.x[i, self.y[i].item() % dim] += signal_strength

        # Add noise
        if noise_level > 0:
            self.x += torch.randn_like(self.x) * noise_level

        # Corrupt some labels if low quality
        if quality < 0.5:
            num_corrupt = int(n * (1 - quality) * 0.3)
            indices = np.random.choice(n, num_corrupt, replace=False)
            for idx in indices:
                self.y[idx] = np.random.randint(0, classes)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx]


class ValuationModel(nn.Module):
    """Simple model for valuation experiments."""

    def __init__(self, config: ValuationConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DataValuator:
    """Data valuation using various methods."""

    def __init__(
        self,
        test_data: ValuationDataset,
        config: ValuationConfig
    ):
        self.test_data = test_data
        self.config = config

    def evaluate_model(self, model: nn.Module) -> float:
        """Evaluate model on test data."""
        model.eval()
        loader = DataLoader(self.test_data, batch_size=64)

        correct, total = 0, 0
        with torch.no_grad():
            for x, y in loader:
                pred = model(x).argmax(dim=1)
                correct += (pred == y).sum().item()
                total += len(y)

        return correct / total

    def train_subset(
        self,
        client_datasets: Dict[int, ValuationDataset],
        client_subset: List[int]
    ) -> float:
        """Train model on subset of clients and return performance."""
        model = ValuationModel(self.config)
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=self.config.learning_rate
        )

        # Combine datasets from subset
        all_x = []
        all_y = []
        for cid in client_subset:
            dataset = client_datasets[cid]
            all_x.append(dataset.x)
            all_y.append(dataset.y)

        if not all_x:
            return self.evaluate_model(model)  # Empty subset

        combined_x = torch.cat(all_x)
        combined_y = torch.cat(all_y)

        # Train
        model.train()
        for _ in range(self.config.local_epochs):
            for i in range(0, len(combined_x), self.config.batch_size):
                x = combined_x[i:i+self.config.batch_size]
                y = combined_y[i:i+self.config.batch_size]

                optimizer.zero_grad()
                loss = F.cross_entropy(model(x), y)
                loss.backward()
                optimizer.step()

        return self.evaluate_model(model)

    def leave_one_out(
        self,
        client_datasets: Dict[int, ValuationDataset]
    ) -> Dict[int, float]:
        """Compute leave-one-out contributions."""
        client_ids = list(client_datasets.keys())

        # Full performance
        full_perf = self.train_subset(client_datasets, client_ids)

        # Leave-one-out
        contributions = {}
        for cid in client_ids:
            subset = [c for c in client_ids if c != cid]
            loo_perf = self.train_subset(client_datasets, subset)
            contributions[cid] = full_perf - loo_perf

        return contributions

    def monte_carlo_shapley(
        self,
        client_datasets: Dict[int, ValuationDataset],
        num_samples: int = 100
    ) -> Dict[int, float]:
        """Compute Shapley values using Monte Carlo sampling."""
        client_ids = list(client_datasets.keys())
        n = len(client_ids)

        shapley = {cid: 0.0 for cid in client_ids}

        for _ in range(num_samples):
            # Random permutation
            perm = np.random.permutation(client_ids)

            prev_perf = self.train_subset(client_datasets, [])
            subset = []

            for cid in perm:
                subset.append(cid)
                curr_perf = self.train_subset(client_datasets, subset)

                # Marginal contribution
                shapley[cid] += (curr_perf - prev_perf) / num_samples
                prev_perf = curr_perf

        return shapley

    def gradient_similarity(
        self,
        client_updates: Dict[int, Dict[str, torch.Tensor]],
        global_model: nn.Module
    ) -> Dict[int, float]:
        """Value based on gradient similarity to optimal."""
        # Compute average gradient direction
        avg_grad = {}
        for cid, update in client_updates.items():
            for key, value in update.items():
                if key not in avg_grad:
                    avg_grad[key] = torch.zeros_like(value)
                avg_grad[key] += value / len(client_updates)

        # Compute similarity scores
        similarities = {}
        for cid, update in client_updates.items():
            sim = 0.0
            norm_update = 0.0
            norm_avg = 0.0

            for key in update:
                flat_update = update[key].flatten()
                flat_avg = avg_grad[key].flatten()

                sim += (flat_update * flat_avg).sum().item()
                norm_update += (flat_update ** 2).sum().item()
                norm_avg += (flat_avg ** 2).sum().item()

            similarities[cid] = sim / (
                np.sqrt(norm_update) * np.sqrt(norm_avg) + 1e-8
            )

        return similarities


class ValuationClient:
    """Client for valuation experiments."""

    def __init__(
        self,
        client_id: int,
        dataset: ValuationDataset,
        config: ValuationConfig
    ):
        self.client_id = client_id
        self.dataset = dataset
        self.config = config

    def train(self, model: nn.Module) -> Dict[str, Any]:
        local = copy.deepcopy(model)
        initial_state = {k: v.clone() for k, v in model.state_dict().items()}

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

        # Compute update (pseudo-gradient)
        update = {}
        for name, param in local.named_parameters():
            update[name] = initial_state[name] - param.data

        return {
            "state_dict": {k: v.cpu() for k, v in local.state_dict().items()},
            "update": update,
            "num_samples": len(self.dataset),
            "avg_loss": total_loss / num_batches,
            "client_id": self.client_id
        }


class ValuationServer:
    """Server with data valuation."""

    def __init__(
        self,
        model: nn.Module,
        clients: List[ValuationClient],
        test_data: ValuationDataset,
        config: ValuationConfig
    ):
        self.model = model
        self.clients = clients
        self.test_data = test_data
        self.config = config

        self.valuator = DataValuator(test_data, config)
        self.client_values: Dict[int, float] = {c.client_id: 0.0 for c in clients}
        self.history: List[Dict] = []

    def aggregate_weighted(
        self,
        updates: List[Dict],
        weights: Optional[Dict[int, float]] = None
    ) -> None:
        """Value-weighted aggregation."""
        if weights is None:
            # Default: sample-weighted
            total_samples = sum(u["num_samples"] for u in updates)
            weights = {u["client_id"]: u["num_samples"] / total_samples for u in updates}

        # Normalize weights
        total_weight = sum(weights.get(u["client_id"], 1) for u in updates)

        new_state = {}
        for key in updates[0]["state_dict"]:
            new_state[key] = sum(
                (weights.get(u["client_id"], 1) / total_weight) *
                u["state_dict"][key].float()
                for u in updates
            )

        self.model.load_state_dict(new_state)

    def train(self) -> List[Dict]:
        """Run training with valuation."""
        logger.info(f"Starting FL with {len(self.clients)} clients")

        for round_num in range(self.config.num_rounds):
            n = min(self.config.clients_per_round, len(self.clients))
            indices = np.random.choice(len(self.clients), n, replace=False)
            selected = [self.clients[i] for i in indices]

            # Collect updates
            updates = [c.train(self.model) for c in selected]

            # Compute gradient-based values
            client_updates = {
                u["client_id"]: u["update"] for u in updates
            }
            grad_values = self.valuator.gradient_similarity(
                client_updates, self.model
            )

            # Update cumulative values
            for cid, val in grad_values.items():
                self.client_values[cid] += val

            # Use quality-weighted aggregation
            weights = {cid: max(0.1, val) for cid, val in grad_values.items()}
            self.aggregate_weighted(updates, weights)

            # Evaluate
            accuracy = self.valuator.evaluate_model(self.model)

            record = {
                "round": round_num,
                "accuracy": accuracy,
                "num_clients": len(updates)
            }
            self.history.append(record)

            if (round_num + 1) % 10 == 0:
                logger.info(f"Round {round_num + 1}: acc={accuracy:.4f}")

        return self.history


def main():
    """Main entry point."""
    print("=" * 60)
    print("Tutorial 140: FL Data Valuation")
    print("=" * 60)

    config = ValuationConfig()
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    # Create clients with varying quality
    clients = []
    qualities = []
    for i in range(config.num_clients):
        quality = 0.3 + 0.7 * (i / (config.num_clients - 1))  # 0.3 to 1.0
        qualities.append(quality)

        dataset = ValuationDataset(
            client_id=i,
            dim=config.input_dim,
            seed=config.seed,
            quality=quality
        )
        client = ValuationClient(i, dataset, config)
        clients.append(client)

    # Test data
    test_data = ValuationDataset(
        client_id=999,
        n=300,
        seed=999,
        quality=1.0
    )

    # Create server
    model = ValuationModel(config)
    server = ValuationServer(model, clients, test_data, config)

    # Train
    history = server.train()

    # Analyze valuations
    print("\n" + "=" * 60)
    print("Client Valuations")
    print("=" * 60)

    # Sort by value
    sorted_values = sorted(
        server.client_values.items(),
        key=lambda x: x[1],
        reverse=True
    )

    for cid, value in sorted_values:
        quality = qualities[cid]
        print(f"Client {cid}: value={value:.4f}, quality={quality:.2f}")

    # Check correlation
    value_arr = [server.client_values[i] for i in range(config.num_clients)]
    corr = np.corrcoef(qualities, value_arr)[0, 1]
    print(f"\nCorrelation between quality and value: {corr:.4f}")

    print("\n" + "=" * 60)
    print(f"Final Accuracy: {history[-1]['accuracy']:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
```

---

## Key Insights

### Valuation Methods Comparison

| Method | Accuracy | Compute | Fairness |
|--------|----------|---------|----------|
| LOO | Moderate | O(n) | Moderate |
| Shapley | High | O(2^n) | High |
| MC Shapley | High | O(m*n) | High |
| Gradient | Moderate | O(1) | Moderate |

### Best Practices

- Use approximate methods for large n
- Combine multiple valuation signals
- Update values incrementally
- Consider temporal contributions

---

## Exercises

1. **Exercise 1**: Implement exact Shapley value
2. **Exercise 2**: Add auction-based pricing
3. **Exercise 3**: Design contribution tokens
4. **Exercise 4**: Visualize value distribution

---

## References

1. Ghorbani, A., & Zou, J. (2019). Data Shapley. In *ICML*.
2. Jia, R., et al. (2019). Towards efficient data valuation. In *NeurIPS*.
3. Song, T., et al. (2020). Profit allocation for FL. In *IEEE TPDS*.

---

*Copyright 2026 Olaf Yunus Laitinen Imanov and Contributors. Released under EUPL 1.2.*
