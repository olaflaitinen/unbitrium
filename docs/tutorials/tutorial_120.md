# Tutorial 120: FL Data Heterogeneity

---

## Metadata

| Property | Value |
|----------|-------|
| **Tutorial ID** | 120 |
| **Title** | Federated Learning Data Heterogeneity |
| **Category** | Core Concepts |
| **Difficulty** | Advanced |
| **Duration** | 120 minutes |
| **Prerequisites** | Tutorial 001-119 |
| **Author** | Unbitrium Contributors |
| **Last Updated** | January 2026 |

---

## Learning Objectives

By completing this tutorial, you will be able to:

1. **Understand** types of data heterogeneity
2. **Implement** heterogeneity-aware FL
3. **Design** algorithms for non-IID data
4. **Analyze** heterogeneity impact
5. **Deploy** robust FL for diverse data

---

## Prerequisites

Before starting this tutorial, ensure you have:

- Completed tutorials 001-119
- Understanding of FL fundamentals
- Knowledge of data distributions
- Familiarity with statistics

---

## Background and Theory

### Types of Heterogeneity

```
Data Heterogeneity:
├── Label Heterogeneity
│   ├── Label skew
│   ├── Class imbalance
│   └── Missing classes
├── Feature Heterogeneity
│   ├── Feature drift
│   ├── Different scales
│   └── Missing features
├── Quantity Heterogeneity
│   ├── Different dataset sizes
│   ├── Sampling bias
│   └── Variable updates
└── Distribution Heterogeneity
    ├── Concept drift
    ├── Prior shift
    └── Covariate shift
```

### Heterogeneity Metrics

| Metric | Description | Range |
|--------|-------------|-------|
| EMD | Earth Mover's Distance | 0-∞ |
| KL Divergence | Distribution difference | 0-∞ |
| Dirichlet α | Concentration parameter | 0-∞ |

---

## Implementation Code

```python
#!/usr/bin/env python3
"""
Tutorial 120: FL Data Heterogeneity

This module implements heterogeneous data scenarios
and algorithms to handle non-IID data.

Copyright 2026 Olaf Yunus Laitinen Imanov and Contributors
Released under EUPL 1.2
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum
import copy
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HeterogeneityType(Enum):
    IID = "iid"
    LABEL_SKEW = "label_skew"
    DIRICHLET = "dirichlet"
    QUANTITY_SKEW = "quantity_skew"


@dataclass
class HeteroConfig:
    """Heterogeneity configuration."""

    num_rounds: int = 50
    num_clients: int = 20
    clients_per_round: int = 10

    input_dim: int = 32
    hidden_dim: int = 64
    num_classes: int = 10

    learning_rate: float = 0.01
    batch_size: int = 32
    local_epochs: int = 3

    # Heterogeneity parameters
    heterogeneity_type: HeterogeneityType = HeterogeneityType.DIRICHLET
    dirichlet_alpha: float = 0.5
    classes_per_client: int = 2

    # Algorithm
    use_scaffold: bool = False
    use_prox: bool = False
    prox_mu: float = 0.01

    seed: int = 42


class HeterogeneousDataGenerator:
    """Generate heterogeneous data distributions."""

    def __init__(self, config: HeteroConfig):
        self.config = config
        self.rng = np.random.RandomState(config.seed)

        # Generate base data
        self.n_total = config.num_clients * 200
        self.x_all = torch.randn(self.n_total, config.input_dim)
        self.y_all = torch.randint(0, config.num_classes, (self.n_total,))

        # Add class signal
        for i in range(self.n_total):
            self.x_all[i, self.y_all[i].item() % config.input_dim] += 2.0

    def generate_iid(self) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Generate IID partitions."""
        indices = self.rng.permutation(self.n_total)
        samples_per_client = self.n_total // self.config.num_clients

        partitions = []
        for i in range(self.config.num_clients):
            start = i * samples_per_client
            end = start + samples_per_client
            idx = indices[start:end]
            partitions.append((self.x_all[idx], self.y_all[idx]))

        return partitions

    def generate_label_skew(self) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Each client has limited classes."""
        partitions = []

        for i in range(self.config.num_clients):
            # Select classes for this client
            classes = self.rng.choice(
                self.config.num_classes,
                self.config.classes_per_client,
                replace=False
            )

            # Get samples from these classes
            mask = torch.zeros(self.n_total, dtype=torch.bool)
            for c in classes:
                mask |= (self.y_all == c)

            idx = torch.where(mask)[0]
            idx = idx[self.rng.choice(len(idx), min(200, len(idx)), replace=False)]

            partitions.append((self.x_all[idx], self.y_all[idx]))

        return partitions

    def generate_dirichlet(self) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Dirichlet distribution over classes."""
        # Sample class proportions for each client
        proportions = self.rng.dirichlet(
            np.ones(self.config.num_classes) * self.config.dirichlet_alpha,
            self.config.num_clients
        )

        # Assign samples based on proportions
        class_indices = {
            c: torch.where(self.y_all == c)[0].numpy()
            for c in range(self.config.num_classes)
        }

        partitions = []
        for i in range(self.config.num_clients):
            client_indices = []

            for c in range(self.config.num_classes):
                n_samples = int(proportions[i, c] * 200)
                if n_samples > 0 and len(class_indices[c]) > 0:
                    idx = self.rng.choice(
                        class_indices[c],
                        min(n_samples, len(class_indices[c])),
                        replace=True
                    )
                    client_indices.extend(idx)

            if len(client_indices) == 0:
                client_indices = self.rng.choice(self.n_total, 50, replace=False)

            client_indices = np.array(client_indices)
            partitions.append((self.x_all[client_indices], self.y_all[client_indices]))

        return partitions

    def generate(self) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Generate based on config."""
        if self.config.heterogeneity_type == HeterogeneityType.IID:
            return self.generate_iid()
        elif self.config.heterogeneity_type == HeterogeneityType.LABEL_SKEW:
            return self.generate_label_skew()
        elif self.config.heterogeneity_type == HeterogeneityType.DIRICHLET:
            return self.generate_dirichlet()

        return self.generate_iid()


class HeteroDataset(Dataset):
    def __init__(self, x: torch.Tensor, y: torch.Tensor):
        self.x = x
        self.y = y

    def __len__(self): return len(self.y)
    def __getitem__(self, idx): return self.x[idx], self.y[idx]


class HeteroModel(nn.Module):
    def __init__(self, config: HeteroConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.num_classes)
        )

    def forward(self, x): return self.net(x)


class HeteroClient:
    def __init__(
        self,
        client_id: int,
        dataset: HeteroDataset,
        config: HeteroConfig
    ):
        self.client_id = client_id
        self.dataset = dataset
        self.config = config

        # For SCAFFOLD
        self.control_variate = None

    def train(
        self,
        model: nn.Module,
        global_model: Optional[nn.Module] = None
    ) -> Dict:
        local = copy.deepcopy(model)
        optimizer = torch.optim.SGD(local.parameters(), lr=self.config.learning_rate)
        loader = DataLoader(self.dataset, batch_size=self.config.batch_size, shuffle=True)

        local.train()
        total_loss, num_batches = 0.0, 0

        for _ in range(self.config.local_epochs):
            for x, y in loader:
                optimizer.zero_grad()
                loss = F.cross_entropy(local(x), y)

                # FedProx regularization
                if self.config.use_prox and global_model:
                    prox_term = 0.0
                    for p_local, p_global in zip(local.parameters(), global_model.parameters()):
                        prox_term += ((p_local - p_global.detach()) ** 2).sum()
                    loss += self.config.prox_mu / 2 * prox_term

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

    def get_class_distribution(self) -> Dict[int, int]:
        """Get local class distribution."""
        dist = {}
        for y in self.dataset.y.tolist():
            dist[y] = dist.get(y, 0) + 1
        return dist


class HeteroServer:
    def __init__(
        self,
        model: nn.Module,
        clients: List[HeteroClient],
        test_data: HeteroDataset,
        config: HeteroConfig
    ):
        self.model = model
        self.clients = clients
        self.test_data = test_data
        self.config = config
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

    def compute_heterogeneity_metric(self) -> float:
        """Compute heterogeneity metric."""
        distributions = [c.get_class_distribution() for c in self.clients]

        # Compute average distribution
        global_dist = {}
        for d in distributions:
            for k, v in d.items():
                global_dist[k] = global_dist.get(k, 0) + v
        total = sum(global_dist.values())
        global_dist = {k: v / total for k, v in global_dist.items()}

        # Compute divergence
        divergences = []
        for d in distributions:
            total_d = sum(d.values())
            local_dist = {k: v / total_d for k, v in d.items()}

            kl = 0.0
            for k in global_dist:
                p = local_dist.get(k, 1e-10)
                q = global_dist[k]
                kl += p * np.log(p / q + 1e-10)
            divergences.append(kl)

        return np.mean(divergences)

    def train(self) -> List[Dict]:
        heterogeneity = self.compute_heterogeneity_metric()
        logger.info(f"Starting FL with heterogeneity metric: {heterogeneity:.4f}")

        for round_num in range(self.config.num_rounds):
            n = min(self.config.clients_per_round, len(self.clients))
            indices = np.random.choice(len(self.clients), n, replace=False)
            selected = [self.clients[i] for i in indices]

            global_model = copy.deepcopy(self.model) if self.config.use_prox else None
            updates = [c.train(self.model, global_model) for c in selected]

            self.aggregate(updates)

            metrics = self.evaluate()

            record = {"round": round_num, **metrics}
            self.history.append(record)

            if (round_num + 1) % 10 == 0:
                logger.info(f"Round {round_num + 1}: acc={metrics['accuracy']:.4f}")

        return self.history


def main():
    print("=" * 60)
    print("Tutorial 120: FL Data Heterogeneity")
    print("=" * 60)

    config = HeteroConfig()
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    # Compare heterogeneity levels
    results = {}

    for htype in HeterogeneityType:
        config.heterogeneity_type = htype

        generator = HeterogeneousDataGenerator(config)
        partitions = generator.generate()

        clients = [
            HeteroClient(i, HeteroDataset(x, y), config)
            for i, (x, y) in enumerate(partitions)
        ]

        test_data = HeteroDataset(generator.x_all[:500], generator.y_all[:500])
        model = HeteroModel(config)

        server = HeteroServer(model, clients, test_data, config)
        history = server.train()

        results[htype.value] = history[-1]["accuracy"]

    print("\n" + "=" * 60)
    print("Heterogeneity Impact")
    for htype, acc in results.items():
        print(f"  {htype}: {acc:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
```

---

## Key Insights

### Handling Heterogeneity

1. **Measure first**: Quantify heterogeneity
2. **Choose algorithm**: FedProx, SCAFFOLD
3. **Personalize**: Local adaptation
4. **Balance**: Trade convergence vs personalization

---

## Exercises

1. **Exercise 1**: Implement SCAFFOLD
2. **Exercise 2**: Add cluster-based FL
3. **Exercise 3**: Design multi-task FL
4. **Exercise 4**: Measure concept drift

---

## References

1. Li, T., et al. (2020). On the convergence of FedAvg on non-IID data. In *ICLR*.
2. Karimireddy, S.P., et al. (2020). SCAFFOLD: Stochastic controlled averaging for FL. In *ICML*.
3. Hsieh, K., et al. (2020). The non-IID data quagmire of decentralized ML. In *ICML*.

---

*Copyright 2026 Olaf Yunus Laitinen Imanov and Contributors. Released under EUPL 1.2.*
