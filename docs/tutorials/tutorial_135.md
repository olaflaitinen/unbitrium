# Tutorial 135: FL Scalability

---

## Metadata

| Property | Value |
|----------|-------|
| **Tutorial ID** | 135 |
| **Title** | Federated Learning Scalability |
| **Category** | Infrastructure |
| **Difficulty** | Advanced |
| **Duration** | 120 minutes |
| **Prerequisites** | Tutorial 001-134 |
| **Author** | Unbitrium Contributors |
| **Last Updated** | January 2026 |

---

## Learning Objectives

By completing this tutorial, you will be able to:

1. **Understand** FL scalability challenges
2. **Implement** systems for thousands of clients
3. **Design** efficient client selection strategies
4. **Analyze** communication bottlenecks
5. **Deploy** production-scale FL systems

---

## Prerequisites

Before starting this tutorial, ensure you have:

- Completed tutorials 001-134
- Understanding of FL fundamentals
- Knowledge of distributed systems
- Familiarity with system design

---

## Background and Theory

### Scalability Challenges in FL

FL systems must scale to handle:
- Millions of clients (e.g., mobile keyboards)
- High-dimensional models
- Limited server resources
- Variable client availability

### Scalability Dimensions

```
FL Scalability Dimensions:
├── Client Scale
│   ├── Client selection
│   ├── Sampling strategies
│   └── Availability patterns
├── Model Scale
│   ├── Large model training
│   ├── Gradient compression
│   └── Partial updates
├── Communication Scale
│   ├── Bandwidth optimization
│   ├── Async aggregation
│   └── Hierarchical FL
└── Compute Scale
    ├── Server parallelization
    ├── Aggregation efficiency
    └── Resource allocation
```

### Scaling Strategies

| Strategy | Clients | Bandwidth | Compute |
|----------|---------|-----------|---------|
| Subsampling | 1% per round | - | - |
| Compression | - | 10-100x | - |
| Hierarchical | - | - | Distributed |
| Async | - | - | Parallel |

---

## Implementation Code

```python
#!/usr/bin/env python3
"""
Tutorial 135: Federated Learning Scalability

This module implements scalable FL with client sampling,
compression, and efficient aggregation.

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
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from enum import Enum

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SelectionStrategy(Enum):
    """Client selection strategies."""
    RANDOM = "random"
    WEIGHTED = "weighted"
    PRIORITIZED = "prioritized"


@dataclass
class ScalabilityConfig:
    """Configuration for scalable FL."""

    # FL parameters
    num_rounds: int = 50
    num_clients: int = 1000
    clients_per_round: int = 50  # 5% sampling

    # Model parameters
    input_dim: int = 32
    hidden_dim: int = 64
    num_classes: int = 10

    # Training parameters
    learning_rate: float = 0.01
    batch_size: int = 32
    local_epochs: int = 2

    # Scalability parameters
    compression_ratio: float = 0.1
    use_async: bool = True
    max_workers: int = 10
    selection_strategy: SelectionStrategy = SelectionStrategy.WEIGHTED

    # Data parameters
    samples_per_client: int = 50

    seed: int = 42


class ScalableDataset(Dataset):
    """Minimal dataset for scalability testing."""

    def __init__(
        self,
        client_id: int,
        n: int = 50,
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


class ScalableModel(nn.Module):
    """Model for scalability testing."""

    def __init__(self, config: ScalabilityConfig):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class GradientCompressor:
    """Compressor for bandwidth efficiency."""

    def __init__(self, ratio: float = 0.1):
        self.ratio = ratio
        self.error_feedback: Dict[str, torch.Tensor] = {}

    def compress(
        self,
        state_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, Any]:
        """Top-k sparsification with error feedback."""
        compressed = {}

        for name, tensor in state_dict.items():
            # Apply error feedback
            if name in self.error_feedback:
                tensor = tensor + self.error_feedback[name]

            flat = tensor.flatten()
            k = max(1, int(len(flat) * self.ratio))

            values, indices = torch.topk(flat.abs(), k)
            selected = flat[indices]

            # Store error
            mask = torch.zeros_like(flat)
            mask[indices] = 1
            self.error_feedback[name] = (flat * (1 - mask)).reshape(tensor.shape)

            compressed[name] = {
                "indices": indices,
                "values": selected,
                "shape": tensor.shape
            }

        return compressed

    def decompress(
        self,
        compressed: Dict[str, Any]
    ) -> Dict[str, torch.Tensor]:
        """Reconstruct state dict."""
        state_dict = {}

        for name, data in compressed.items():
            flat = torch.zeros(int(np.prod(data["shape"])))
            flat[data["indices"]] = data["values"]
            state_dict[name] = flat.reshape(data["shape"])

        return state_dict


class ClientSelector:
    """Client selection strategies for scale."""

    def __init__(
        self,
        strategy: SelectionStrategy,
        num_clients: int
    ):
        self.strategy = strategy
        self.num_clients = num_clients

        # Track client statistics
        self.client_stats: Dict[int, Dict] = {
            i: {"rounds": 0, "loss": 1.0, "samples": 0}
            for i in range(num_clients)
        }

    def update_stats(self, client_id: int, loss: float, samples: int) -> None:
        """Update client statistics."""
        self.client_stats[client_id]["rounds"] += 1
        self.client_stats[client_id]["loss"] = loss
        self.client_stats[client_id]["samples"] = samples

    def select(self, k: int) -> List[int]:
        """Select k clients."""
        if self.strategy == SelectionStrategy.RANDOM:
            return list(np.random.choice(self.num_clients, k, replace=False))

        elif self.strategy == SelectionStrategy.WEIGHTED:
            # Weight by data size
            weights = np.array([
                self.client_stats[i].get("samples", 1) + 1
                for i in range(self.num_clients)
            ])
            weights = weights / weights.sum()
            return list(np.random.choice(
                self.num_clients, k, replace=False, p=weights
            ))

        elif self.strategy == SelectionStrategy.PRIORITIZED:
            # Prioritize high-loss clients
            losses = [
                self.client_stats[i].get("loss", 1.0)
                for i in range(self.num_clients)
            ]
            weights = np.array(losses) + 0.1
            weights = weights / weights.sum()
            return list(np.random.choice(
                self.num_clients, k, replace=False, p=weights
            ))

        return list(np.random.choice(self.num_clients, k, replace=False))


class ScalableClient:
    """Lightweight client for scale testing."""

    def __init__(
        self,
        client_id: int,
        dataset: ScalableDataset,
        config: ScalabilityConfig
    ):
        self.client_id = client_id
        self.dataset = dataset
        self.config = config
        self.compressor = GradientCompressor(config.compression_ratio)

    def train(
        self,
        model_state: Dict[str, torch.Tensor]
    ) -> Dict[str, Any]:
        """Train and return compressed update."""

        # Create local model
        local_model = ScalableModel(self.config)
        local_model.load_state_dict(model_state)

        optimizer = torch.optim.SGD(
            local_model.parameters(),
            lr=self.config.learning_rate
        )

        loader = DataLoader(
            self.dataset,
            batch_size=self.config.batch_size,
            shuffle=True
        )

        local_model.train()
        total_loss = 0.0
        num_batches = 0

        for _ in range(self.config.local_epochs):
            for x, y in loader:
                optimizer.zero_grad()
                loss = F.cross_entropy(local_model(x), y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(local_model.parameters(), 1.0)
                optimizer.step()

                total_loss += loss.item()
                num_batches += 1

        # Compress update
        update = {
            k: v.cpu() for k, v in local_model.state_dict().items()
        }
        compressed = self.compressor.compress(update)

        return {
            "compressed": compressed,
            "num_samples": len(self.dataset),
            "avg_loss": total_loss / num_batches,
            "client_id": self.client_id
        }


class ScalableServer:
    """Server designed for scale."""

    def __init__(
        self,
        model: nn.Module,
        clients: List[ScalableClient],
        test_data: ScalableDataset,
        config: ScalabilityConfig
    ):
        self.model = model
        self.clients = clients
        self.test_data = test_data
        self.config = config

        self.selector = ClientSelector(
            config.selection_strategy,
            len(clients)
        )
        self.compressor = GradientCompressor(config.compression_ratio)
        self.history: List[Dict] = []

    def _train_client(
        self,
        client: ScalableClient,
        model_state: Dict[str, torch.Tensor]
    ) -> Dict[str, Any]:
        """Train single client (for parallel execution)."""
        return client.train(model_state)

    def aggregate(self, updates: List[Dict]) -> None:
        """Aggregate compressed updates."""
        if not updates:
            return

        total_samples = sum(u["num_samples"] for u in updates)
        new_state = None

        for u in updates:
            decompressed = self.compressor.decompress(u["compressed"])
            weight = u["num_samples"] / total_samples

            if new_state is None:
                new_state = {k: v * weight for k, v in decompressed.items()}
            else:
                for k, v in decompressed.items():
                    new_state[k] += v * weight

        if new_state:
            self.model.load_state_dict(new_state)

    def evaluate(self) -> Dict[str, float]:
        """Evaluate model."""
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
        """Run scalable FL training."""
        logger.info(
            f"Starting FL with {len(self.clients)} clients, "
            f"{self.config.clients_per_round} per round"
        )

        for round_num in range(self.config.num_rounds):
            start_time = time.time()

            # Select clients
            selected_ids = self.selector.select(self.config.clients_per_round)
            selected_clients = [self.clients[i] for i in selected_ids]

            # Get current model state
            model_state = {
                k: v.clone() for k, v in self.model.state_dict().items()
            }

            # Train in parallel if async enabled
            updates = []
            if self.config.use_async:
                with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
                    futures = {
                        executor.submit(
                            self._train_client, client, model_state
                        ): client
                        for client in selected_clients
                    }

                    for future in as_completed(futures):
                        try:
                            update = future.result()
                            updates.append(update)

                            # Update client stats
                            self.selector.update_stats(
                                update["client_id"],
                                update["avg_loss"],
                                update["num_samples"]
                            )
                        except Exception as e:
                            logger.error(f"Client failed: {e}")
            else:
                for client in selected_clients:
                    update = client.train(model_state)
                    updates.append(update)
                    self.selector.update_stats(
                        update["client_id"],
                        update["avg_loss"],
                        update["num_samples"]
                    )

            # Aggregate
            self.aggregate(updates)

            # Evaluate
            metrics = self.evaluate()
            round_time = time.time() - start_time

            record = {
                "round": round_num,
                **metrics,
                "num_clients": len(updates),
                "round_time_s": round_time,
                "avg_loss": np.mean([u["avg_loss"] for u in updates])
            }
            self.history.append(record)

            if (round_num + 1) % 10 == 0:
                logger.info(
                    f"Round {round_num + 1}: "
                    f"acc={metrics['accuracy']:.4f}, "
                    f"time={round_time:.2f}s, "
                    f"clients={len(updates)}"
                )

        return self.history


def main():
    """Main entry point."""
    print("=" * 60)
    print("Tutorial 135: FL Scalability")
    print("=" * 60)

    config = ScalabilityConfig()
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    # Create many clients
    logger.info(f"Creating {config.num_clients} clients...")
    clients = []
    for i in range(config.num_clients):
        dataset = ScalableDataset(
            client_id=i,
            n=config.samples_per_client,
            dim=config.input_dim,
            seed=config.seed
        )
        client = ScalableClient(i, dataset, config)
        clients.append(client)

    # Test data
    test_data = ScalableDataset(
        client_id=999999,
        n=500,
        seed=999999
    )

    # Model
    model = ScalableModel(config)
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Train
    server = ScalableServer(model, clients, test_data, config)
    history = server.train()

    # Summary
    avg_time = np.mean([r["round_time_s"] for r in history])
    print("\n" + "=" * 60)
    print("Training Complete")
    print(f"Final Accuracy: {history[-1]['accuracy']:.4f}")
    print(f"Average Round Time: {avg_time:.2f}s")
    print(f"Total Clients: {config.num_clients}")
    print(f"Clients per Round: {config.clients_per_round}")
    print("=" * 60)


if __name__ == "__main__":
    main()
```

---

## Key Insights

### Scalability Best Practices

1. **Client Sampling**: Sample 1-10% of clients per round
2. **Compression**: Use top-k sparsification with error feedback
3. **Parallelization**: Train clients in parallel
4. **Smart Selection**: Use weighted or prioritized selection

### Performance Tips

- Use lightweight client objects
- Minimize state copying
- Profile and optimize bottlenecks
- Consider hierarchical aggregation

---

## Exercises

1. **Exercise 1**: Implement client caching
2. **Exercise 2**: Add adaptive sampling rates
3. **Exercise 3**: Implement hierarchical FL
4. **Exercise 4**: Profile and optimize for 100K clients

---

## References

1. Bonawitz, K., et al. (2019). Towards FL at scale. In *MLSys*.
2. Li, T., et al. (2020). Federated optimization for heterogeneous networks. In *MLSys*.
3. Kairouz, P., et al. (2021). Advances and open problems in FL. *FnTML*.

---

*Copyright 2026 Olaf Yunus Laitinen Imanov and Contributors. Released under EUPL 1.2.*
