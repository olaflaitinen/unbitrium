# Tutorial 124: FL Model Debugging

---

## Metadata

| Property | Value |
|----------|-------|
| **Tutorial ID** | 124 |
| **Title** | Federated Learning Model Debugging |
| **Category** | Engineering |
| **Difficulty** | Advanced |
| **Duration** | 90 minutes |
| **Prerequisites** | Tutorial 001-123 |
| **Author** | Unbitrium Contributors |
| **Last Updated** | January 2026 |

---

## Learning Objectives

By completing this tutorial, you will be able to:

1. **Understand** debugging challenges in FL
2. **Implement** distributed debugging tools
3. **Diagnose** model convergence issues
4. **Analyze** client-specific problems
5. **Deploy** robust debugging pipelines

---

## Prerequisites

Before starting this tutorial, ensure you have:

- Completed tutorials 001-123
- Understanding of FL fundamentals
- Knowledge of model debugging
- Familiarity with distributed systems

---

## Background and Theory

### FL Debugging Challenges

FL debugging is harder than centralized:
- Can't inspect client data
- Aggregated updates hide issues
- Client heterogeneity complicates analysis
- Distributed execution is harder to trace

### Common Issues

```
FL Issues:
├── Convergence
│   ├── Divergence
│   ├── Oscillation
│   └── Slow progress
├── Client Issues
│   ├── Stale updates
│   ├── Byzantine behavior
│   └── Data quality
├── Aggregation
│   ├── Weight explosion
│   ├── NaN gradients
│   └── Bias
└── System
    ├── Communication failures
    ├── Timeouts
    └── Resource exhaustion
```

---

## Implementation Code

```python
#!/usr/bin/env python3
"""
Tutorial 124: Federated Learning Model Debugging

This module implements debugging tools for FL including
convergence analysis and issue detection.

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
class DebugConfig:
    """Configuration for debugging FL."""

    num_rounds: int = 30
    num_clients: int = 10
    clients_per_round: int = 5

    input_dim: int = 32
    hidden_dim: int = 64
    num_classes: int = 10

    learning_rate: float = 0.01
    batch_size: int = 32
    local_epochs: int = 3

    seed: int = 42


class DebugMonitor:
    """Monitor and diagnose FL training."""

    def __init__(self):
        self.round_metrics: List[Dict] = []
        self.client_metrics: Dict[int, List[Dict]] = {}
        self.gradient_norms: List[Dict] = []
        self.weight_stats: List[Dict] = []
        self.issues_detected: List[Dict] = []

    def log_round(
        self,
        round_num: int,
        accuracy: float,
        loss: float,
        client_updates: List[Dict]
    ) -> None:
        """Log round metrics."""
        self.round_metrics.append({
            "round": round_num,
            "accuracy": accuracy,
            "loss": loss,
            "num_clients": len(client_updates)
        })

        # Check for issues
        self._check_convergence()

    def log_client(
        self,
        client_id: int,
        round_num: int,
        metrics: Dict
    ) -> None:
        """Log client-specific metrics."""
        if client_id not in self.client_metrics:
            self.client_metrics[client_id] = []

        self.client_metrics[client_id].append({
            "round": round_num,
            **metrics
        })

    def log_gradients(
        self,
        round_num: int,
        client_id: int,
        gradient_stats: Dict
    ) -> None:
        """Log gradient statistics."""
        self.gradient_norms.append({
            "round": round_num,
            "client_id": client_id,
            **gradient_stats
        })

        # Check for gradient issues
        self._check_gradients(gradient_stats, client_id, round_num)

    def log_weights(
        self,
        round_num: int,
        weight_stats: Dict
    ) -> None:
        """Log weight statistics."""
        self.weight_stats.append({
            "round": round_num,
            **weight_stats
        })

        self._check_weights(weight_stats, round_num)

    def _check_convergence(self) -> None:
        """Check for convergence issues."""
        if len(self.round_metrics) < 5:
            return

        recent = self.round_metrics[-5:]
        losses = [m["loss"] for m in recent]

        # Check for divergence
        if losses[-1] > losses[0] * 2:
            self.issues_detected.append({
                "type": "DIVERGENCE",
                "round": recent[-1]["round"],
                "message": f"Loss increased from {losses[0]:.4f} to {losses[-1]:.4f}"
            })

        # Check for oscillation
        if all(abs(losses[i] - losses[i-1]) / losses[i-1] > 0.2 for i in range(1, len(losses))):
            self.issues_detected.append({
                "type": "OSCILLATION",
                "round": recent[-1]["round"],
                "message": "Loss is oscillating significantly"
            })

    def _check_gradients(
        self,
        stats: Dict,
        client_id: int,
        round_num: int
    ) -> None:
        """Check for gradient issues."""
        if stats.get("has_nan", False):
            self.issues_detected.append({
                "type": "NAN_GRADIENT",
                "round": round_num,
                "client_id": client_id,
                "message": "NaN detected in gradients"
            })

        if stats.get("max_norm", 0) > 1000:
            self.issues_detected.append({
                "type": "EXPLODING_GRADIENT",
                "round": round_num,
                "client_id": client_id,
                "message": f"Gradient norm: {stats['max_norm']:.2f}"
            })

    def _check_weights(self, stats: Dict, round_num: int) -> None:
        """Check for weight issues."""
        if stats.get("has_nan", False):
            self.issues_detected.append({
                "type": "NAN_WEIGHTS",
                "round": round_num,
                "message": "NaN detected in model weights"
            })

    def get_summary(self) -> Dict[str, Any]:
        """Get debugging summary."""
        if not self.round_metrics:
            return {}

        return {
            "total_rounds": len(self.round_metrics),
            "final_accuracy": self.round_metrics[-1]["accuracy"],
            "final_loss": self.round_metrics[-1]["loss"],
            "num_issues": len(self.issues_detected),
            "issue_types": list(set(i["type"] for i in self.issues_detected))
        }

    def print_issues(self) -> None:
        """Print detected issues."""
        if not self.issues_detected:
            print("No issues detected!")
            return

        print(f"\n{len(self.issues_detected)} issues detected:")
        for issue in self.issues_detected:
            print(f"  [{issue['type']}] Round {issue['round']}: {issue['message']}")


class DebugDataset(Dataset):
    """Dataset for debugging experiments."""

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


class DebugModel(nn.Module):
    """Model with debugging hooks."""

    def __init__(self, config: DebugConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def get_weight_stats(self) -> Dict[str, Any]:
        """Get statistics about weights."""
        stats = {
            "has_nan": False,
            "has_inf": False,
            "max_weight": 0.0,
            "min_weight": 0.0
        }

        for param in self.parameters():
            if torch.isnan(param).any():
                stats["has_nan"] = True
            if torch.isinf(param).any():
                stats["has_inf"] = True

            stats["max_weight"] = max(stats["max_weight"], param.abs().max().item())
            stats["min_weight"] = min(stats["min_weight"], param.min().item())

        return stats


class DebugClient:
    """Client with debugging capabilities."""

    def __init__(
        self,
        client_id: int,
        dataset: DebugDataset,
        config: DebugConfig,
        monitor: DebugMonitor
    ):
        self.client_id = client_id
        self.dataset = dataset
        self.config = config
        self.monitor = monitor

    def train(self, model: nn.Module, round_num: int) -> Dict[str, Any]:
        """Train with debugging."""
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
        gradient_norms = []

        for _ in range(self.config.local_epochs):
            for x, y in loader:
                optimizer.zero_grad()
                output = local(x)
                loss = F.cross_entropy(output, y)
                loss.backward()

                # Collect gradient info
                grad_norm = 0.0
                has_nan = False
                for param in local.parameters():
                    if param.grad is not None:
                        grad_norm += param.grad.norm().item() ** 2
                        if torch.isnan(param.grad).any():
                            has_nan = True

                gradient_norms.append(grad_norm ** 0.5)

                optimizer.step()

                total_loss += loss.item()
                num_batches += 1

        # Log gradient stats
        grad_stats = {
            "mean_norm": np.mean(gradient_norms),
            "max_norm": np.max(gradient_norms),
            "has_nan": has_nan
        }
        self.monitor.log_gradients(round_num, self.client_id, grad_stats)

        # Log client metrics
        client_metrics = {
            "avg_loss": total_loss / num_batches,
            "num_samples": len(self.dataset)
        }
        self.monitor.log_client(self.client_id, round_num, client_metrics)

        return {
            "state_dict": {k: v.cpu() for k, v in local.state_dict().items()},
            "num_samples": len(self.dataset),
            "avg_loss": total_loss / num_batches,
            "client_id": self.client_id
        }


class DebugServer:
    """Server with debugging."""

    def __init__(
        self,
        model: nn.Module,
        clients: List[DebugClient],
        test_data: DebugDataset,
        config: DebugConfig,
        monitor: DebugMonitor
    ):
        self.model = model
        self.clients = clients
        self.test_data = test_data
        self.config = config
        self.monitor = monitor
        self.history: List[Dict] = []

    def aggregate(self, updates: List[Dict]) -> None:
        """Aggregate updates."""
        total_samples = sum(u["num_samples"] for u in updates)
        new_state = {}

        for key in updates[0]["state_dict"]:
            new_state[key] = sum(
                (u["num_samples"] / total_samples) * u["state_dict"][key].float()
                for u in updates
            )

        self.model.load_state_dict(new_state)

    def evaluate(self) -> Tuple[float, float]:
        """Evaluate model."""
        self.model.eval()
        loader = DataLoader(self.test_data, batch_size=64)

        correct, total = 0, 0
        total_loss = 0.0

        with torch.no_grad():
            for x, y in loader:
                output = self.model(x)
                pred = output.argmax(dim=1)
                correct += (pred == y).sum().item()
                total += len(y)
                total_loss += F.cross_entropy(output, y).item()

        return correct / total, total_loss / len(loader)

    def train(self) -> List[Dict]:
        """Run training with debugging."""
        logger.info(f"Starting FL with debugging ({len(self.clients)} clients)")

        for round_num in range(self.config.num_rounds):
            n = min(self.config.clients_per_round, len(self.clients))
            indices = np.random.choice(len(self.clients), n, replace=False)
            selected = [self.clients[i] for i in indices]

            updates = [c.train(self.model, round_num) for c in selected]

            self.aggregate(updates)

            # Log weight stats
            weight_stats = self.model.get_weight_stats()
            self.monitor.log_weights(round_num, weight_stats)

            # Evaluate
            accuracy, loss = self.evaluate()

            # Log round
            self.monitor.log_round(round_num, accuracy, loss, updates)

            record = {"round": round_num, "accuracy": accuracy, "loss": loss}
            self.history.append(record)

            if (round_num + 1) % 10 == 0:
                logger.info(f"Round {round_num + 1}: acc={accuracy:.4f}, loss={loss:.4f}")

        return self.history


def main():
    """Main entry point."""
    print("=" * 60)
    print("Tutorial 124: FL Model Debugging")
    print("=" * 60)

    config = DebugConfig()
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    monitor = DebugMonitor()

    clients = []
    for i in range(config.num_clients):
        dataset = DebugDataset(client_id=i, dim=config.input_dim, seed=config.seed)
        client = DebugClient(i, dataset, config, monitor)
        clients.append(client)

    test_data = DebugDataset(client_id=999, n=300, seed=999)
    model = DebugModel(config)

    server = DebugServer(model, clients, test_data, config, monitor)
    history = server.train()

    print("\n" + "=" * 60)
    print("Debugging Summary")
    print("=" * 60)

    summary = monitor.get_summary()
    for key, value in summary.items():
        print(f"  {key}: {value}")

    monitor.print_issues()
    print("=" * 60)


if __name__ == "__main__":
    main()
```

---

## Key Insights

### Debugging Best Practices

1. **Monitor gradients**: Catch issues early
2. **Track per-client**: Find problematic clients
3. **Log convergence**: Detect divergence
4. **Validate weights**: Check for NaN/Inf

---

## Exercises

1. **Exercise 1**: Add visualization tools
2. **Exercise 2**: Implement anomaly detection
3. **Exercise 3**: Design auto-correction
4. **Exercise 4**: Add replay debugging

---

## References

1. PyTorch debugging documentation
2. Li, T., et al. (2020). Federated optimization. *MLSys*.
3. Kairouz, P., et al. (2021). Advances and open problems in FL. *FnTML*.

---

*Copyright 2026 Olaf Yunus Laitinen Imanov and Contributors. Released under EUPL 1.2.*
