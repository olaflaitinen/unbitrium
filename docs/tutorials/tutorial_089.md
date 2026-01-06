# Tutorial 089: FL Learning Rate Scheduling

---

## Metadata

| Property | Value |
|----------|-------|
| **Tutorial ID** | 089 |
| **Title** | FL Learning Rate Scheduling |
| **Category** | Optimization |
| **Difficulty** | Intermediate |
| **Duration** | 90 minutes |
| **Prerequisites** | Tutorial 001-088 |
| **Author** | Unbitrium Contributors |
| **Last Updated** | January 2026 |

---

## Learning Objectives

By the end of this tutorial, you will be able to:

1. **Understand** learning rate scheduling in federated settings.
2. **Implement** various decay strategies (step, cosine, exponential).
3. **Design** warmup schedules for stable training.
4. **Analyze** scheduling impact on convergence.
5. **Apply** client and server-side scheduling.
6. **Evaluate** adaptive scheduling strategies.
7. **Create** custom scheduling policies.

---

## Prerequisites

- **Completed Tutorials**: 001-088
- **Knowledge**: Learning rate schedules, optimization
- **Libraries**: PyTorch, NumPy

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Callable
from torch.utils.data import Dataset, DataLoader
import copy
import math

print(f"PyTorch: {torch.__version__}")
```

---

## Background and Theory

### Learning Rate Schedules

| Schedule | Formula | Properties |
|----------|---------|------------|
| Constant | η | Baseline |
| Step | η × γ^(t/step) | Discrete drops |
| Exponential | η × γ^t | Smooth decay |
| Cosine | η_min + (η-η_min)(1+cos(πt/T))/2 | Smooth, cyclic |
| Linear | η × (1 - t/T) | Simple decay |
| Polynomial | η × (1 - t/T)^p | Flexible |

### Warmup Strategies

| Strategy | Description | Use Case |
|----------|-------------|----------|
| Linear | LR increases linearly | Most common |
| Exponential | LR increases exponentially | Fast warmup |
| Gradual | Very slow increase | Sensitive models |

### FL Scheduling Considerations

```mermaid
graph TB
    subgraph "Server Schedule"
        SLR[Server LR η_s(t)]
        SWARM[Warmup Phase]
        SDECAY[Decay Phase]
    end

    subgraph "Client Schedule"
        CLR[Client LR η_c(t)]
        CWARM[Local Warmup]
        CSTABLE[Training Phase]
    end

    subgraph "Combined"
        ROUND[Round t]
        EPOCH[Local Epoch e]
    end

    ROUND --> SLR
    ROUND --> CLR
    EPOCH --> CLR
    SWARM --> SDECAY
    CWARM --> CSTABLE
```

---

## Implementation Code

### Part 1: Scheduler Implementation

```python
#!/usr/bin/env python3
"""
Tutorial 089: FL Learning Rate Scheduling

Comprehensive implementation of learning rate scheduling
for federated learning with warmup and decay strategies.

Author: Unbitrium Contributors
License: EUPL-1.2
"""

from __future__ import annotations
import copy
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Callable, Optional
from enum import Enum
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


class ScheduleType(Enum):
    """Types of learning rate schedules."""
    CONSTANT = "constant"
    STEP = "step"
    EXPONENTIAL = "exponential"
    COSINE = "cosine"
    LINEAR = "linear"
    POLYNOMIAL = "polynomial"
    CYCLIC = "cyclic"


class WarmupType(Enum):
    """Types of warmup strategies."""
    NONE = "none"
    LINEAR = "linear"
    EXPONENTIAL = "exponential"
    GRADUAL = "gradual"


@dataclass
class ScheduleConfig:
    """Configuration for LR scheduling."""
    
    # General
    num_rounds: int = 100
    num_clients: int = 20
    clients_per_round: int = 10
    local_epochs: int = 5
    batch_size: int = 32
    seed: int = 42
    
    # Learning rates
    initial_lr: float = 0.1
    min_lr: float = 0.001
    
    # Server schedule
    server_schedule: ScheduleType = ScheduleType.COSINE
    server_warmup: WarmupType = WarmupType.LINEAR
    server_warmup_rounds: int = 10
    
    # Client schedule
    client_schedule: ScheduleType = ScheduleType.CONSTANT
    client_warmup: WarmupType = WarmupType.NONE
    
    # Schedule parameters
    step_size: int = 20
    gamma: float = 0.5
    polynomial_power: float = 2.0
    
    # Model
    input_dim: int = 32
    hidden_dim: int = 128
    num_classes: int = 10


class LearningRateScheduler:
    """Comprehensive learning rate scheduler."""
    
    def __init__(
        self,
        config: ScheduleConfig,
        schedule_type: ScheduleType,
        warmup_type: WarmupType,
        warmup_steps: int = 0,
        total_steps: int = 100,
    ):
        self.config = config
        self.schedule_type = schedule_type
        self.warmup_type = warmup_type
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.current_step = 0
    
    def get_warmup_factor(self, step: int) -> float:
        """Get warmup multiplier."""
        if step >= self.warmup_steps or self.warmup_type == WarmupType.NONE:
            return 1.0
        
        progress = step / self.warmup_steps
        
        if self.warmup_type == WarmupType.LINEAR:
            return progress
        elif self.warmup_type == WarmupType.EXPONENTIAL:
            return 2.0 ** (progress * 10 - 10)  # Starts near 0
        elif self.warmup_type == WarmupType.GRADUAL:
            return progress ** 2  # Slower initial increase
        else:
            return 1.0
    
    def get_decay_lr(self, step: int) -> float:
        """Get decayed learning rate after warmup."""
        # Adjust step for warmup
        decay_step = step - self.warmup_steps
        decay_total = self.total_steps - self.warmup_steps
        
        if decay_step < 0:
            decay_step = 0
        
        progress = decay_step / max(decay_total, 1)
        
        initial = self.config.initial_lr
        min_lr = self.config.min_lr
        
        if self.schedule_type == ScheduleType.CONSTANT:
            return initial
        
        elif self.schedule_type == ScheduleType.STEP:
            num_drops = decay_step // self.config.step_size
            return initial * (self.config.gamma ** num_drops)
        
        elif self.schedule_type == ScheduleType.EXPONENTIAL:
            return initial * (self.config.gamma ** decay_step)
        
        elif self.schedule_type == ScheduleType.COSINE:
            return min_lr + 0.5 * (initial - min_lr) * (1 + math.cos(math.pi * progress))
        
        elif self.schedule_type == ScheduleType.LINEAR:
            return initial - (initial - min_lr) * progress
        
        elif self.schedule_type == ScheduleType.POLYNOMIAL:
            return min_lr + (initial - min_lr) * ((1 - progress) ** self.config.polynomial_power)
        
        elif self.schedule_type == ScheduleType.CYCLIC:
            cycle_steps = self.config.step_size
            cycle_progress = (decay_step % cycle_steps) / cycle_steps
            return min_lr + 0.5 * (initial - min_lr) * (1 + math.cos(math.pi * cycle_progress))
        
        else:
            return initial
    
    def get_lr(self, step: Optional[int] = None) -> float:
        """Get learning rate for given step."""
        if step is None:
            step = self.current_step
        
        warmup_factor = self.get_warmup_factor(step)
        base_lr = self.get_decay_lr(step)
        
        return base_lr * warmup_factor
    
    def step(self) -> float:
        """Advance scheduler and return current LR."""
        lr = self.get_lr(self.current_step)
        self.current_step += 1
        return lr
    
    def reset(self) -> None:
        """Reset scheduler to initial state."""
        self.current_step = 0


class SchedulerDataset(Dataset):
    """Dataset for scheduler experiments."""
    
    def __init__(self, features: np.ndarray, labels: np.ndarray):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self): return len(self.labels)
    def __getitem__(self, idx): return self.features[idx], self.labels[idx]


class SchedulerModel(nn.Module):
    """Model for scheduler experiments."""
    
    def __init__(self, config: ScheduleConfig):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(config.hidden_dim // 2, config.num_classes),
        )
    
    def forward(self, x): return self.network(x)
```

### Part 2: FL System with Scheduling

```python
def create_data(config: ScheduleConfig) -> Tuple[List[SchedulerDataset], SchedulerDataset]:
    """Create datasets."""
    np.random.seed(config.seed)
    
    # Training data
    datasets = []
    for i in range(config.num_clients):
        n = np.random.randint(80, 120)
        x = np.random.randn(n, config.input_dim).astype(np.float32)
        y = np.random.randint(0, config.num_classes, n)
        for j in range(n):
            x[j, y[j] % config.input_dim] += 2.0
        datasets.append(SchedulerDataset(x, y))
    
    # Test data
    test_x = np.random.randn(500, config.input_dim).astype(np.float32)
    test_y = np.random.randint(0, config.num_classes, 500)
    for j in range(500):
        test_x[j, test_y[j] % config.input_dim] += 2.0
    
    return datasets, SchedulerDataset(test_x, test_y)


class ScheduledClient:
    """FL client with learning rate scheduling."""
    
    def __init__(self, client_id: int, dataset: SchedulerDataset, config: ScheduleConfig):
        self.client_id = client_id
        self.dataset = dataset
        self.config = config
    
    def train(self, model: nn.Module, lr: float) -> Dict[str, Any]:
        """Train with given learning rate."""
        local = copy.deepcopy(model)
        opt = torch.optim.SGD(local.parameters(), lr=lr, momentum=0.9)
        loader = DataLoader(self.dataset, batch_size=self.config.batch_size, shuffle=True)
        
        local.train()
        total_loss, n_batches = 0, 0
        
        for _ in range(self.config.local_epochs):
            for x, y in loader:
                opt.zero_grad()
                loss = F.cross_entropy(local(x), y)
                loss.backward()
                opt.step()
                total_loss += loss.item()
                n_batches += 1
        
        return {
            "state_dict": {k: v.cpu() for k, v in local.state_dict().items()},
            "num_samples": len(self.dataset),
            "loss": total_loss / n_batches if n_batches > 0 else 0,
        }


class ScheduledServer:
    """FL server with learning rate scheduling."""
    
    def __init__(self, model: nn.Module, clients: List[ScheduledClient],
                 test_dataset: SchedulerDataset, config: ScheduleConfig):
        self.model = model
        self.clients = clients
        self.test_dataset = test_dataset
        self.config = config
        self.history = []
        
        # Server scheduler
        self.scheduler = LearningRateScheduler(
            config,
            config.server_schedule,
            config.server_warmup,
            config.server_warmup_rounds,
            config.num_rounds,
        )
    
    def aggregate(self, updates: List[Dict]) -> None:
        total = sum(u["num_samples"] for u in updates)
        new_state = {}
        for key in self.model.state_dict():
            new_state[key] = sum((u["num_samples"]/total) * u["state_dict"][key].float() for u in updates)
        self.model.load_state_dict(new_state)
    
    def evaluate(self) -> Tuple[float, float]:
        self.model.eval()
        loader = DataLoader(self.test_dataset, batch_size=128)
        correct, total, loss = 0, 0, 0.0
        
        with torch.no_grad():
            for x, y in loader:
                out = self.model(x)
                loss += F.cross_entropy(out, y).item() * len(y)
                correct += (out.argmax(1) == y).sum().item()
                total += len(y)
        
        return correct / total, loss / total
    
    def train(self) -> List[Dict]:
        for r in range(self.config.num_rounds):
            # Get current learning rate
            current_lr = self.scheduler.step()
            
            # Select and train clients
            selected = np.random.choice(self.clients, min(self.config.clients_per_round, len(self.clients)), replace=False)
            updates = [c.train(self.model, current_lr) for c in selected]
            
            self.aggregate(updates)
            acc, loss = self.evaluate()
            
            self.history.append({
                "round": r,
                "accuracy": acc,
                "loss": loss,
                "lr": current_lr,
            })
            
            if (r + 1) % 10 == 0:
                print(f"Round {r+1}: acc={acc:.4f}, lr={current_lr:.6f}")
        
        return self.history


def compare_schedules():
    """Compare different scheduling strategies."""
    results = {}
    
    for schedule in [ScheduleType.CONSTANT, ScheduleType.COSINE, ScheduleType.STEP]:
        config = ScheduleConfig(num_rounds=50, num_clients=10, server_schedule=schedule)
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        
        datasets, test = create_data(config)
        clients = [ScheduledClient(i, d, config) for i, d in enumerate(datasets)]
        model = SchedulerModel(config)
        server = ScheduledServer(model, clients, test, config)
        
        print(f"\n=== {schedule.value.upper()} ===")
        history = server.train()
        results[schedule.value] = history[-1]["accuracy"]
    
    print("\n=== Results ===")
    for name, acc in results.items():
        print(f"{name}: {acc:.4f}")


if __name__ == "__main__":
    compare_schedules()
```

---

## Exercises

1. **Exercise 1**: Implement one-cycle learning rate.
2. **Exercise 2**: Add learning rate finder.
3. **Exercise 3**: Create round-adaptive scheduling.
4. **Exercise 4**: Add restart-based schedules.
5. **Exercise 5**: Implement per-layer learning rates.

---

## References

1. Li, T., et al. (2020). On the convergence of FedAvg on non-IID data. In *ICLR*.
2. Loshchilov, I., & Hutter, F. (2017). SGDR: Stochastic gradient descent with warm restarts. In *ICLR*.
3. Smith, L.N. (2017). Cyclical learning rates for training neural networks. In *WACV*.
4. Gotmare, A., et al. (2019). A closer look at deep learning heuristics. In *ICLR*.
5. Wang, J., et al. (2021). Cooperative SGD: A unified framework for FL. In *ICML*.

---

*Copyright 2026 Olaf Yunus Laitinen Imanov and Contributors. Released under EUPL 1.2.*
