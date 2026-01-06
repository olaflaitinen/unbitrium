# Tutorial 134: FL Profiling

---

## Metadata

| Property | Value |
|----------|-------|
| **Tutorial ID** | 134 |
| **Title** | Federated Learning Profiling |
| **Category** | Engineering |
| **Difficulty** | Advanced |
| **Duration** | 90 minutes |
| **Prerequisites** | Tutorial 001-133 |
| **Author** | Unbitrium Contributors |
| **Last Updated** | January 2026 |

---

## Learning Objectives

By completing this tutorial, you will be able to:

1. **Understand** FL performance bottlenecks
2. **Implement** profiling tools for FL
3. **Analyze** compute, memory, and communication
4. **Optimize** FL system performance
5. **Deploy** instrumented FL pipelines

---

## Prerequisites

Before starting this tutorial, ensure you have:

- Completed tutorials 001-133
- Understanding of FL fundamentals
- Knowledge of performance profiling
- Familiarity with system metrics

---

## Background and Theory

### What to Profile in FL

```
FL Profiling Dimensions:
├── Compute
│   ├── Training time per epoch
│   ├── Aggregation overhead
│   └── GPU/CPU utilization
├── Memory
│   ├── Model size
│   ├── Gradient buffers
│   └── Peak memory usage
├── Communication
│   ├── Upload/download time
│   ├── Bandwidth usage
│   └── Compression ratios
└── System
    ├── Round time breakdown
    ├── Client availability
    └── Failure rates
```

### Profiling Goals

| Metric | Goal | Action |
|--------|------|--------|
| Round time | Minimize | Async, compression |
| Memory | Fit device | Pruning, quantization |
| Bandwidth | Reduce | Compression |
| Accuracy | Maintain | Balance tradeoffs |

---

## Implementation Code

```python
#!/usr/bin/env python3
"""
Tutorial 134: Federated Learning Profiling

This module implements profiling for FL systems to identify
bottlenecks and optimize performance.

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
from contextlib import contextmanager
import copy
import logging
import time
import tracemalloc

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ProfileConfig:
    """Configuration for profiling FL."""

    num_rounds: int = 20
    num_clients: int = 10
    clients_per_round: int = 5

    input_dim: int = 32
    hidden_dim: int = 64
    num_classes: int = 10

    learning_rate: float = 0.01
    batch_size: int = 32
    local_epochs: int = 3

    seed: int = 42


class Timer:
    """High-resolution timer for profiling."""

    def __init__(self):
        self.start_time: float = 0
        self.end_time: float = 0
        self.elapsed: float = 0

    def start(self) -> None:
        self.start_time = time.perf_counter()

    def stop(self) -> float:
        self.end_time = time.perf_counter()
        self.elapsed = self.end_time - self.start_time
        return self.elapsed

    @contextmanager
    def measure(self):
        self.start()
        yield
        self.stop()


class MemoryProfiler:
    """Memory usage profiler."""

    def __init__(self):
        self.peak_memory: int = 0
        self.current_memory: int = 0

    def start(self) -> None:
        tracemalloc.start()

    def stop(self) -> Tuple[int, int]:
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        self.current_memory = current
        self.peak_memory = peak
        return current, peak

    @contextmanager
    def measure(self):
        self.start()
        yield
        self.stop()


class FLProfiler:
    """Comprehensive FL profiler."""

    def __init__(self):
        self.timer = Timer()
        self.memory = MemoryProfiler()

        self.round_metrics: List[Dict] = []
        self.client_metrics: Dict[int, List[Dict]] = {}
        self.aggregation_times: List[float] = []
        self.communication_bytes: List[int] = []

    def profile_round(
        self,
        round_num: int,
        client_updates: List[Dict],
        aggregation_time: float,
        total_time: float
    ) -> Dict[str, Any]:
        """Profile a training round."""
        # Communication estimate
        total_bytes = sum(
            self._estimate_update_size(u)
            for u in client_updates
        )

        # Training times
        training_times = [
            u.get("training_time", 0) for u in client_updates
        ]

        metrics = {
            "round": round_num,
            "total_time_s": total_time,
            "aggregation_time_s": aggregation_time,
            "training_time_s": {
                "mean": np.mean(training_times),
                "max": np.max(training_times),
                "min": np.min(training_times)
            },
            "num_clients": len(client_updates),
            "total_bytes": total_bytes,
            "bytes_per_client": total_bytes / len(client_updates) if client_updates else 0
        }

        self.round_metrics.append(metrics)
        self.aggregation_times.append(aggregation_time)
        self.communication_bytes.append(total_bytes)

        return metrics

    def profile_client(
        self,
        client_id: int,
        training_time: float,
        num_samples: int,
        memory_bytes: int
    ) -> Dict[str, Any]:
        """Profile client training."""
        metrics = {
            "training_time_s": training_time,
            "num_samples": num_samples,
            "samples_per_second": num_samples / training_time if training_time > 0 else 0,
            "memory_bytes": memory_bytes
        }

        if client_id not in self.client_metrics:
            self.client_metrics[client_id] = []
        self.client_metrics[client_id].append(metrics)

        return metrics

    def _estimate_update_size(self, update: Dict) -> int:
        """Estimate update size in bytes."""
        total = 0
        for key, value in update.get("state_dict", {}).items():
            if isinstance(value, torch.Tensor):
                total += value.numel() * value.element_size()
        return total

    def get_summary(self) -> Dict[str, Any]:
        """Get profiling summary."""
        if not self.round_metrics:
            return {}

        total_times = [m["total_time_s"] for m in self.round_metrics]

        return {
            "total_rounds": len(self.round_metrics),
            "avg_round_time_s": np.mean(total_times),
            "total_training_time_s": sum(total_times),
            "avg_aggregation_time_s": np.mean(self.aggregation_times),
            "total_bytes_transferred": sum(self.communication_bytes),
            "avg_bytes_per_round": np.mean(self.communication_bytes),
            "client_count": len(self.client_metrics)
        }

    def print_summary(self) -> None:
        """Print formatted summary."""
        summary = self.get_summary()

        print("\n" + "=" * 50)
        print("PROFILING SUMMARY")
        print("=" * 50)

        for key, value in summary.items():
            if isinstance(value, float):
                print(f"{key}: {value:.4f}")
            else:
                print(f"{key}: {value}")


class ProfileDataset(Dataset):
    """Dataset for profiling experiments."""

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


class ProfileModel(nn.Module):
    """Model for profiling."""

    def __init__(self, config: ProfileConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def get_size_bytes(self) -> int:
        """Get model size."""
        return sum(p.numel() * p.element_size() for p in self.parameters())


class ProfileClient:
    """Client with profiling."""

    def __init__(
        self,
        client_id: int,
        dataset: ProfileDataset,
        config: ProfileConfig,
        profiler: FLProfiler
    ):
        self.client_id = client_id
        self.dataset = dataset
        self.config = config
        self.profiler = profiler

    def train(self, model: nn.Module) -> Dict[str, Any]:
        """Train with profiling."""
        timer = Timer()
        memory = MemoryProfiler()

        timer.start()
        memory.start()

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

        current_mem, peak_mem = memory.stop()
        training_time = timer.stop()

        # Profile
        self.profiler.profile_client(
            self.client_id,
            training_time,
            len(self.dataset),
            peak_mem
        )

        return {
            "state_dict": {k: v.cpu() for k, v in local.state_dict().items()},
            "num_samples": len(self.dataset),
            "avg_loss": total_loss / num_batches,
            "client_id": self.client_id,
            "training_time": training_time,
            "peak_memory": peak_mem
        }


class ProfileServer:
    """Server with profiling."""

    def __init__(
        self,
        model: nn.Module,
        clients: List[ProfileClient],
        test_data: ProfileDataset,
        config: ProfileConfig,
        profiler: FLProfiler
    ):
        self.model = model
        self.clients = clients
        self.test_data = test_data
        self.config = config
        self.profiler = profiler
        self.history: List[Dict] = []

    def aggregate(self, updates: List[Dict]) -> float:
        """Aggregate with timing."""
        timer = Timer()
        timer.start()

        total_samples = sum(u["num_samples"] for u in updates)
        new_state = {}

        for key in updates[0]["state_dict"]:
            new_state[key] = sum(
                (u["num_samples"] / total_samples) * u["state_dict"][key].float()
                for u in updates
            )

        self.model.load_state_dict(new_state)

        return timer.stop()

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
        """Run profiled FL training."""
        logger.info(f"Starting profiled FL with {len(self.clients)} clients")

        for round_num in range(self.config.num_rounds):
            round_timer = Timer()
            round_timer.start()

            # Select clients
            n = min(self.config.clients_per_round, len(self.clients))
            indices = np.random.choice(len(self.clients), n, replace=False)
            selected = [self.clients[i] for i in indices]

            # Collect updates
            updates = [c.train(self.model) for c in selected]

            # Aggregate
            agg_time = self.aggregate(updates)

            total_time = round_timer.stop()

            # Profile round
            self.profiler.profile_round(
                round_num, updates, agg_time, total_time
            )

            # Evaluate
            metrics = self.evaluate()

            record = {
                "round": round_num,
                **metrics,
                "total_time": total_time
            }
            self.history.append(record)

            if (round_num + 1) % 5 == 0:
                logger.info(
                    f"Round {round_num + 1}: "
                    f"acc={metrics['accuracy']:.4f}, "
                    f"time={total_time:.3f}s"
                )

        return self.history


def main():
    """Main entry point."""
    print("=" * 60)
    print("Tutorial 134: FL Profiling")
    print("=" * 60)

    config = ProfileConfig()
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    # Create profiler
    profiler = FLProfiler()

    # Create clients
    clients = []
    for i in range(config.num_clients):
        dataset = ProfileDataset(client_id=i, dim=config.input_dim, seed=config.seed)
        client = ProfileClient(i, dataset, config, profiler)
        clients.append(client)

    test_data = ProfileDataset(client_id=999, n=300, seed=999)
    model = ProfileModel(config)

    logger.info(f"Model size: {model.get_size_bytes():,} bytes")

    # Train
    server = ProfileServer(model, clients, test_data, config, profiler)
    history = server.train()

    # Print summary
    profiler.print_summary()

    print(f"\nFinal Accuracy: {history[-1]['accuracy']:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
```

---

## Key Insights

### Profiling Tips

1. **Measure all phases**: Training, communication, aggregation
2. **Track per-client**: Identify stragglers
3. **Monitor memory**: Prevent OOM
4. **Estimate bandwidth**: Plan compression

---

## Exercises

1. **Exercise 1**: Add GPU profiling
2. **Exercise 2**: Create visualization dashboard
3. **Exercise 3**: Implement bottleneck detection
4. **Exercise 4**: Add comparative analysis

---

## References

1. PyTorch Profiler documentation
2. Li, T., et al. (2020). Federated optimization. *MLSys*.
3. Bonawitz, K., et al. (2019). Towards FL at scale. In *MLSys*.

---

*Copyright 2026 Olaf Yunus Laitinen Imanov and Contributors. Released under EUPL 1.2.*
