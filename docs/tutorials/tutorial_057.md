# Tutorial 057: FL Continual Learning

---

## Metadata

| Property | Value |
|----------|-------|
| **Tutorial ID** | 057 |
| **Title** | Federated Learning Continual Learning |
| **Category** | Advanced Techniques |
| **Difficulty** | Expert |
| **Duration** | 120 minutes |
| **Prerequisites** | Tutorial 001-056 |
| **Author** | Unbitrium Contributors |
| **Last Updated** | January 2026 |

---

## Learning Objectives

By completing this tutorial, you will be able to:

1. **Understand** continual learning in FL
2. **Implement** catastrophic forgetting prevention
3. **Design** task-incremental FL
4. **Analyze** knowledge retention
5. **Deploy** lifelong FL systems

---

## Prerequisites

Before starting this tutorial, ensure you have:

- Completed tutorials 001-056
- Understanding of FL fundamentals
- Knowledge of continual learning
- Familiarity with EWC, rehearsal

---

## Background and Theory

### Continual Learning in FL

```
FL Continual Learning:
├── Challenges
│   ├── Catastrophic forgetting
│   ├── Task interference
│   └── Concept drift
├── Approaches
│   ├── EWC (Elastic Weight Consolidation)
│   ├── Rehearsal/replay
│   ├── Progressive networks
│   └── PackNet
├── Federated Context
│   ├── Client task sequences
│   ├── Global knowledge retention
│   └── Personalized memories
└── Scenarios
    ├── Task-incremental
    ├── Class-incremental
    └── Domain-incremental
```

---

## Implementation Code

```python
#!/usr/bin/env python3
"""
Tutorial 057: FL Continual Learning

This module implements continual learning for
federated learning with forgetting prevention.

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
import copy
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ContinualConfig:
    """Continual learning configuration."""

    num_tasks: int = 3
    rounds_per_task: int = 20
    num_clients: int = 10
    clients_per_round: int = 5

    input_dim: int = 32
    hidden_dim: int = 64
    classes_per_task: int = 3

    learning_rate: float = 0.01
    batch_size: int = 32
    local_epochs: int = 3

    ewc_lambda: float = 100.0
    memory_size: int = 100

    seed: int = 42


class ContinualModel(nn.Module):
    """Model with EWC support."""

    def __init__(self, config: ContinualConfig):
        super().__init__()
        total_classes = config.num_tasks * config.classes_per_task

        self.net = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, total_classes)
        )

        # EWC importance weights
        self.fisher: Dict[str, torch.Tensor] = {}
        self.old_params: Dict[str, torch.Tensor] = {}

    def forward(self, x): return self.net(x)

    def update_fisher(self, data_loader: DataLoader) -> None:
        """Compute Fisher information matrix."""
        self.fisher = {k: torch.zeros_like(p) for k, p in self.named_parameters()}
        self.old_params = {k: p.clone().detach() for k, p in self.named_parameters()}

        self.train()
        for x, y in data_loader:
            self.zero_grad()
            output = self(x)
            loss = F.cross_entropy(output, y)
            loss.backward()

            for k, p in self.named_parameters():
                if p.grad is not None:
                    self.fisher[k] += p.grad.data ** 2

        # Normalize
        n = len(data_loader.dataset)
        for k in self.fisher:
            self.fisher[k] /= n

    def ewc_loss(self) -> torch.Tensor:
        """Compute EWC regularization loss."""
        loss = 0.0
        for k, p in self.named_parameters():
            if k in self.fisher:
                loss += (self.fisher[k] * (p - self.old_params[k]) ** 2).sum()
        return loss


class TaskDataset(Dataset):
    """Dataset for a specific task."""

    def __init__(
        self,
        task_id: int,
        n: int = 200,
        dim: int = 32,
        classes_per_task: int = 3,
        seed: int = 0
    ):
        np.random.seed(seed + task_id)
        self.x = torch.randn(n, dim, dtype=torch.float32)

        # Labels offset by task
        base_class = task_id * classes_per_task
        self.y = torch.randint(base_class, base_class + classes_per_task, (n,), dtype=torch.long)

        for i in range(n):
            self.x[i, self.y[i].item() % dim] += 2.0

    def __len__(self): return len(self.y)
    def __getitem__(self, idx): return self.x[idx], self.y[idx]


class MemoryBuffer:
    """Rehearsal memory buffer."""

    def __init__(self, max_size: int):
        self.max_size = max_size
        self.x_mem: List[torch.Tensor] = []
        self.y_mem: List[torch.Tensor] = []

    def add(self, x: torch.Tensor, y: torch.Tensor) -> None:
        """Add samples to memory."""
        for i in range(len(y)):
            if len(self.x_mem) < self.max_size:
                self.x_mem.append(x[i].clone())
                self.y_mem.append(y[i].clone())
            else:
                # Random replacement
                idx = np.random.randint(0, self.max_size)
                self.x_mem[idx] = x[i].clone()
                self.y_mem[idx] = y[i].clone()

    def sample(self, n: int) -> Optional[tuple]:
        """Sample from memory."""
        if len(self.x_mem) == 0:
            return None

        n = min(n, len(self.x_mem))
        indices = np.random.choice(len(self.x_mem), n, replace=False)

        x_batch = torch.stack([self.x_mem[i] for i in indices])
        y_batch = torch.stack([self.y_mem[i] for i in indices])

        return x_batch, y_batch


class ContinualClient:
    """Client with continual learning."""

    def __init__(self, client_id: int, config: ContinualConfig):
        self.client_id = client_id
        self.config = config
        self.memory = MemoryBuffer(config.memory_size)
        self.current_task = 0

    def set_task(self, task_id: int, dataset: TaskDataset) -> None:
        self.current_task = task_id
        self.dataset = dataset

    def train(self, model: nn.Module) -> Dict:
        local = copy.deepcopy(model)
        optimizer = torch.optim.Adam(local.parameters(), lr=self.config.learning_rate)
        loader = DataLoader(self.dataset, batch_size=self.config.batch_size, shuffle=True)

        local.train()
        total_loss = 0.0
        num_batches = 0

        for _ in range(self.config.local_epochs):
            for x, y in loader:
                optimizer.zero_grad()

                loss = F.cross_entropy(local(x), y)

                # EWC regularization
                if local.fisher:
                    loss += self.config.ewc_lambda * local.ewc_loss()

                # Rehearsal
                mem_sample = self.memory.sample(self.config.batch_size // 2)
                if mem_sample:
                    x_mem, y_mem = mem_sample
                    loss += F.cross_entropy(local(x_mem), y_mem)

                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                num_batches += 1

                # Add to memory
                self.memory.add(x, y)

        return {
            "state_dict": {k: v.cpu() for k, v in local.state_dict().items()},
            "num_samples": len(self.dataset),
            "avg_loss": total_loss / num_batches
        }


class ContinualServer:
    """Server for continual FL."""

    def __init__(self, model: nn.Module, clients: List[ContinualClient], config: ContinualConfig):
        self.model = model
        self.clients = clients
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

    def evaluate(self, test_datasets: List[TaskDataset]) -> Dict[str, float]:
        """Evaluate on all tasks."""
        self.model.eval()

        task_accs = []
        for task_id, dataset in enumerate(test_datasets):
            loader = DataLoader(dataset, batch_size=64)
            correct, total = 0, 0
            with torch.no_grad():
                for x, y in loader:
                    pred = self.model(x).argmax(dim=1)
                    correct += (pred == y).sum().item()
                    total += len(y)
            task_accs.append(correct / total)

        return {
            "mean_accuracy": np.mean(task_accs),
            "per_task": task_accs
        }

    def train(self) -> List[Dict]:
        logger.info("Starting continual FL")

        test_datasets = [TaskDataset(t, n=100, seed=999 + t) for t in range(self.config.num_tasks)]

        for task_id in range(self.config.num_tasks):
            logger.info(f"Task {task_id + 1}/{self.config.num_tasks}")

            # Set client datasets for this task
            for i, client in enumerate(self.clients):
                dataset = TaskDataset(task_id, seed=self.config.seed + i * 100 + task_id)
                client.set_task(task_id, dataset)

            for round_num in range(self.config.rounds_per_task):
                n = min(self.config.clients_per_round, len(self.clients))
                indices = np.random.choice(len(self.clients), n, replace=False)
                selected = [self.clients[i] for i in indices]

                updates = [c.train(self.model) for c in selected]
                self.aggregate(updates)

            # Update Fisher after task
            loader = DataLoader(test_datasets[task_id], batch_size=32)
            self.model.update_fisher(loader)

            metrics = self.evaluate(test_datasets[:task_id + 1])
            self.history.append({"task": task_id, **metrics})
            logger.info(f"Task {task_id + 1} complete: mean_acc={metrics['mean_accuracy']:.4f}")

        return self.history


def main():
    print("=" * 60)
    print("Tutorial 057: FL Continual Learning")
    print("=" * 60)

    config = ContinualConfig()
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    clients = [ContinualClient(i, config) for i in range(config.num_clients)]
    model = ContinualModel(config)

    server = ContinualServer(model, clients, config)
    history = server.train()

    print("\n" + "=" * 60)
    print("Continual FL Complete")
    print(f"Final mean accuracy: {history[-1]['mean_accuracy']:.4f}")
    print(f"Per-task: {history[-1]['per_task']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
```

---

## Key Insights

### Continual Learning Best Practices

1. **EWC**: Protect important weights
2. **Rehearsal**: Store representative samples
3. **Task boundaries**: Know when tasks change
4. **Evaluate all**: Track forgetting

---

## Exercises

1. **Exercise 1**: Implement PackNet
2. **Exercise 2**: Add progressive networks
3. **Exercise 3**: Design task detection
4. **Exercise 4**: Add generative replay

---

## References

1. Kirkpatrick, J., et al. (2017). Overcoming catastrophic forgetting. *PNAS*.
2. Yoon, J., et al. (2021). Federated continual learning. In *ICML*.
3. Rolnick, D., et al. (2019). Experience replay for continual learning. In *NeurIPS*.

---

*Copyright 2026 Olaf Yunus Laitinen Imanov and Contributors. Released under EUPL 1.2.*
