# Tutorial 106: FL Async Training

---

## Metadata

| Property | Value |
|----------|-------|
| **Tutorial ID** | 106 |
| **Title** | FL Asynchronous Training |
| **Category** | Systems |
| **Difficulty** | Advanced |
| **Duration** | 90 minutes |
| **Prerequisites** | Tutorial 001-105 |
| **Author** | Unbitrium Contributors |
| **Last Updated** | January 2026 |

---

## Learning Objectives

By completing this tutorial, you will be able to:

1. **Understand** async FL concepts
2. **Implement** async aggregation
3. **Design** staleness handling
4. **Analyze** convergence
5. **Deploy** async FL systems

---

## Background and Theory

### Async FL Architecture

```
Async FL:
├── Update Handling
│   ├── Immediate aggregation
│   ├── Buffered updates
│   └── Priority queues
├── Staleness
│   ├── Age of update
│   ├── Staleness weighting
│   └── Bounding staleness
├── Algorithms
│   ├── FedAsync
│   ├── Async-FedAvg
│   └── Bounded-staleness SGD
└── Benefits
    ├── No synchronization barriers
    ├── Better resource utilization
    └── Fault tolerance
```

---

## Implementation Code

```python
#!/usr/bin/env python3
"""
Tutorial 106: FL Asynchronous Training

This module implements asynchronous federated
learning without synchronization barriers.

Copyright 2026 Olaf Yunus Laitinen Imanov and Contributors
Released under EUPL 1.2
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass
from typing import Dict, List, Optional
from queue import PriorityQueue
import copy
import logging
import threading
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class AsyncConfig:
    """Async FL configuration."""

    num_updates: int = 200
    num_clients: int = 20

    input_dim: int = 32
    hidden_dim: int = 64
    num_classes: int = 10

    learning_rate: float = 0.01
    batch_size: int = 32
    local_epochs: int = 2

    # Async params
    max_staleness: int = 10
    staleness_weight_decay: float = 0.5

    seed: int = 42


class Update:
    """Client update with staleness info."""

    def __init__(
        self,
        state_dict: Dict[str, torch.Tensor],
        client_id: int,
        base_version: int,
        num_samples: int
    ):
        self.state_dict = state_dict
        self.client_id = client_id
        self.base_version = base_version
        self.num_samples = num_samples
        self.timestamp = time.time()

    def staleness(self, current_version: int) -> int:
        return current_version - self.base_version


class AsyncDataset(Dataset):
    def __init__(self, n: int = 200, dim: int = 32, classes: int = 10, seed: int = 0):
        np.random.seed(seed)
        self.x = torch.randn(n, dim, dtype=torch.float32)
        self.y = torch.randint(0, classes, (n,), dtype=torch.long)
        for i in range(n):
            self.x[i, self.y[i].item() % dim] += 2.0

    def __len__(self): return len(self.y)
    def __getitem__(self, idx): return self.x[idx], self.y[idx]


class AsyncModel(nn.Module):
    def __init__(self, config: AsyncConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.num_classes)
        )

    def forward(self, x): return self.net(x)


class AsyncClient:
    """Async FL client."""

    def __init__(self, client_id: int, dataset: AsyncDataset, config: AsyncConfig):
        self.client_id = client_id
        self.dataset = dataset
        self.config = config

        # Simulate variable training time
        self.speed_factor = np.random.uniform(0.5, 2.0)

    def train(self, model: nn.Module, current_version: int) -> Update:
        """Train and return update."""
        # Simulate latency
        time.sleep(0.001 * self.speed_factor)

        local = copy.deepcopy(model)
        optimizer = torch.optim.SGD(local.parameters(), lr=self.config.learning_rate)
        loader = DataLoader(self.dataset, batch_size=self.config.batch_size, shuffle=True)

        local.train()
        for _ in range(self.config.local_epochs):
            for x, y in loader:
                optimizer.zero_grad()
                loss = F.cross_entropy(local(x), y)
                loss.backward()
                optimizer.step()

        return Update(
            state_dict={k: v.cpu() for k, v in local.state_dict().items()},
            client_id=self.client_id,
            base_version=current_version,
            num_samples=len(self.dataset)
        )


class AsyncAggregator:
    """Async update aggregator."""

    def __init__(self, config: AsyncConfig):
        self.config = config

    def compute_weight(self, update: Update, current_version: int) -> float:
        """Compute update weight based on staleness."""
        staleness = update.staleness(current_version)

        if staleness > self.config.max_staleness:
            return 0.0

        # Exponential decay with staleness
        weight = (self.config.staleness_weight_decay ** staleness)
        return weight * update.num_samples

    def aggregate(
        self,
        model_state: Dict[str, torch.Tensor],
        update: Update,
        current_version: int
    ) -> Dict[str, torch.Tensor]:
        """Aggregate single update into model."""
        weight = self.compute_weight(update, current_version)

        if weight <= 0:
            return model_state

        # Mixing rate
        alpha = weight / (weight + 100)  # Damping factor

        new_state = {}
        for key in model_state:
            new_state[key] = (1 - alpha) * model_state[key] + alpha * update.state_dict[key]

        return new_state


class AsyncServer:
    """Async FL server."""

    def __init__(self, model: nn.Module, clients: List[AsyncClient], test_data: AsyncDataset, config: AsyncConfig):
        self.model = model
        self.clients = clients
        self.test_data = test_data
        self.config = config

        self.aggregator = AsyncAggregator(config)
        self.version = 0
        self.history: List[Dict] = []

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
        logger.info("Starting async FL")

        updates_processed = 0

        while updates_processed < self.config.num_updates:
            # Randomly select a client to complete
            client = np.random.choice(self.clients)

            update = client.train(self.model, self.version)

            # Aggregate update
            new_state = self.aggregator.aggregate(
                self.model.state_dict(),
                update,
                self.version
            )
            self.model.load_state_dict(new_state)

            self.version += 1
            updates_processed += 1

            # Log periodically
            if updates_processed % 20 == 0:
                metrics = self.evaluate()
                staleness = update.staleness(self.version)

                record = {
                    "update": updates_processed,
                    "version": self.version,
                    "staleness": staleness,
                    **metrics
                }
                self.history.append(record)

                logger.info(f"Update {updates_processed}: acc={metrics['accuracy']:.4f}")

        return self.history


def main():
    print("=" * 60)
    print("Tutorial 106: FL Async Training")
    print("=" * 60)

    config = AsyncConfig()
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    clients = [AsyncClient(i, AsyncDataset(seed=config.seed + i), config) for i in range(config.num_clients)]
    test_data = AsyncDataset(seed=999)
    model = AsyncModel(config)

    server = AsyncServer(model, clients, test_data, config)
    history = server.train()

    print("\n" + "=" * 60)
    print("Async Training Complete")
    print(f"Final accuracy: {history[-1]['accuracy']:.4f}")
    print(f"Total versions: {history[-1]['version']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
```

---

## Key Insights

### Async Best Practices

1. **Bound staleness**: Reject too-old updates
2. **Weight by staleness**: Fresher = higher weight
3. **Damping**: Smooth aggregation
4. **Monitor convergence**: Track stability

---

## Exercises

1. **Exercise 1**: Add bounded staleness
2. **Exercise 2**: Implement priority scheduling
3. **Exercise 3**: Design adaptive mixing
4. **Exercise 4**: Compare sync vs async

---

## References

1. Xie, C., et al. (2020). Asynchronous federated optimization. *arXiv*.
2. Stich, S.U. (2019). Local SGD converges fast. In *ICLR*.
3. Nguyen, J., et al. (2022). Federated learning with buffered asynchronous aggregation. In *AISTATS*.

---

*Copyright 2026 Olaf Yunus Laitinen Imanov and Contributors. Released under EUPL 1.2.*
