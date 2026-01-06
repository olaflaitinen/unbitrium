# Tutorial 056: FL Meta-Learning

---

## Metadata

| Property | Value |
|----------|-------|
| **Tutorial ID** | 056 |
| **Title** | Federated Learning Meta-Learning |
| **Category** | Advanced Techniques |
| **Difficulty** | Expert |
| **Duration** | 120 minutes |
| **Prerequisites** | Tutorial 001-055 |
| **Author** | Unbitrium Contributors |
| **Last Updated** | January 2026 |

---

## Learning Objectives

By completing this tutorial, you will be able to:

1. **Understand** meta-learning in FL
2. **Implement** MAML-based FL
3. **Design** personalized FL systems
4. **Analyze** adaptation efficiency
5. **Deploy** meta-learning FL

---

## Prerequisites

Before starting this tutorial, ensure you have:

- Completed tutorials 001-055
- Understanding of FL fundamentals
- Knowledge of meta-learning
- Familiarity with MAML

---

## Background and Theory

### Meta-Learning for FL

```
FL Meta-Learning:
├── MAML-based
│   ├── Per-FedAvg
│   ├── Reptile
│   └── First-order MAML
├── Personalization
│   ├── Local adaptation
│   ├── Few-shot learning
│   └── Task-specific heads
├── Algorithms
│   ├── Meta-SGD
│   ├── ProtoNets
│   └── Matching networks
└── Applications
    ├── Quick adaptation
    ├── New client onboarding
    └── Domain adaptation
```

---

## Implementation Code

```python
#!/usr/bin/env python3
"""
Tutorial 056: FL Meta-Learning

This module implements meta-learning for
federated learning personalization.

Copyright 2026 Olaf Yunus Laitinen Imanov and Contributors
Released under EUPL 1.2
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass
from typing import Dict, List, Tuple
import copy
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class MetaConfig:
    """Meta-learning configuration."""
    
    num_rounds: int = 50
    num_clients: int = 20
    clients_per_round: int = 10
    
    input_dim: int = 32
    hidden_dim: int = 64
    num_classes: int = 10
    
    outer_lr: float = 0.01
    inner_lr: float = 0.1
    inner_steps: int = 5
    batch_size: int = 16
    
    adaptation_steps: int = 3
    
    seed: int = 42


class MetaModel(nn.Module):
    """Model for meta-learning."""
    
    def __init__(self, config: MetaConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.num_classes)
        )
    
    def forward(self, x): return self.net(x)
    
    def clone(self) -> 'MetaModel':
        return copy.deepcopy(self)
    
    def adapt(
        self,
        support_x: torch.Tensor,
        support_y: torch.Tensor,
        inner_lr: float,
        steps: int
    ) -> 'MetaModel':
        """Fast adaptation on support set."""
        adapted = self.clone()
        
        for _ in range(steps):
            loss = F.cross_entropy(adapted(support_x), support_y)
            grads = torch.autograd.grad(loss, adapted.parameters())
            
            # Manual gradient update
            for param, grad in zip(adapted.parameters(), grads):
                param.data = param.data - inner_lr * grad
        
        return adapted


class MetaDataset(Dataset):
    """Dataset for meta-learning tasks."""
    
    def __init__(
        self,
        n: int = 200,
        dim: int = 32,
        classes: int = 10,
        seed: int = 0,
        task_shift: float = 0.0
    ):
        np.random.seed(seed)
        self.x = torch.randn(n, dim, dtype=torch.float32) + task_shift
        self.y = torch.randint(0, classes, (n,), dtype=torch.long)
        for i in range(n):
            self.x[i, self.y[i].item() % dim] += 2.0
    
    def __len__(self): return len(self.y)
    def __getitem__(self, idx): return self.x[idx], self.y[idx]
    
    def get_support_query(
        self,
        k_shot: int = 5
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Split into support and query sets."""
        perm = torch.randperm(len(self))
        support_idx = perm[:k_shot * 10]  # 10 classes * k_shot
        query_idx = perm[k_shot * 10:]
        
        return (
            self.x[support_idx], self.y[support_idx],
            self.x[query_idx], self.y[query_idx]
        )


class MAMLClient:
    """Client using MAML for personalization."""
    
    def __init__(
        self,
        client_id: int,
        dataset: MetaDataset,
        config: MetaConfig
    ):
        self.client_id = client_id
        self.dataset = dataset
        self.config = config
    
    def compute_meta_gradient(self, model: nn.Module) -> Dict:
        """Compute MAML-style meta-gradient."""
        support_x, support_y, query_x, query_y = self.dataset.get_support_query()
        
        # Inner loop: adapt on support
        adapted = model.adapt(
            support_x, support_y,
            self.config.inner_lr,
            self.config.inner_steps
        )
        
        # Outer loss: evaluate on query
        query_loss = F.cross_entropy(adapted(query_x), query_y)
        
        # Compute meta-gradient
        meta_grads = torch.autograd.grad(query_loss, model.parameters())
        
        return {
            "grads": {k: g.cpu() for k, g in zip(model.state_dict().keys(), meta_grads)},
            "num_samples": len(self.dataset),
            "query_loss": query_loss.item()
        }
    
    def evaluate_adapted(self, model: nn.Module) -> float:
        """Evaluate after adaptation."""
        support_x, support_y, query_x, query_y = self.dataset.get_support_query()
        
        adapted = model.adapt(
            support_x, support_y,
            self.config.inner_lr,
            self.config.adaptation_steps
        )
        
        adapted.eval()
        with torch.no_grad():
            pred = adapted(query_x).argmax(dim=1)
            accuracy = (pred == query_y).float().mean().item()
        
        return accuracy


class MAMLServer:
    """Server for MAML-based FL."""
    
    def __init__(
        self,
        model: nn.Module,
        clients: List[MAMLClient],
        test_clients: List[MAMLClient],
        config: MetaConfig
    ):
        self.model = model
        self.clients = clients
        self.test_clients = test_clients
        self.config = config
        self.optimizer = torch.optim.Adam(model.parameters(), lr=config.outer_lr)
        self.history: List[Dict] = []
    
    def aggregate_gradients(self, updates: List[Dict]) -> None:
        """Aggregate meta-gradients and update."""
        self.optimizer.zero_grad()
        
        total = sum(u["num_samples"] for u in updates)
        
        # Average gradients
        for param, key in zip(self.model.parameters(), self.model.state_dict().keys()):
            param.grad = sum(
                (u["num_samples"] / total) * u["grads"][key].float()
                for u in updates
            )
        
        self.optimizer.step()
    
    def evaluate(self) -> Dict[str, float]:
        """Evaluate on test clients after adaptation."""
        accuracies = [c.evaluate_adapted(self.model) for c in self.test_clients]
        return {"adapted_accuracy": np.mean(accuracies)}
    
    def train(self) -> List[Dict]:
        logger.info("Starting MAML-based FL")
        
        for round_num in range(self.config.num_rounds):
            n = min(self.config.clients_per_round, len(self.clients))
            indices = np.random.choice(len(self.clients), n, replace=False)
            selected = [self.clients[i] for i in indices]
            
            updates = [c.compute_meta_gradient(self.model) for c in selected]
            self.aggregate_gradients(updates)
            
            metrics = self.evaluate()
            
            record = {"round": round_num, **metrics}
            self.history.append(record)
            
            if (round_num + 1) % 10 == 0:
                logger.info(f"Round {round_num + 1}: adapted_acc={metrics['adapted_accuracy']:.4f}")
        
        return self.history


def main():
    print("=" * 60)
    print("Tutorial 056: FL Meta-Learning")
    print("=" * 60)
    
    config = MetaConfig()
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    
    # Training clients with different task shifts
    clients = [
        MAMLClient(i, MetaDataset(seed=config.seed + i, task_shift=i * 0.1), config)
        for i in range(config.num_clients)
    ]
    
    # Test clients with new shifts
    test_clients = [
        MAMLClient(100 + i, MetaDataset(seed=999 + i, task_shift=i * 0.2), config)
        for i in range(5)
    ]
    
    model = MetaModel(config)
    
    server = MAMLServer(model, clients, test_clients, config)
    history = server.train()
    
    print("\n" + "=" * 60)
    print("Meta-Learning FL Complete")
    print(f"Final adapted accuracy: {history[-1]['adapted_accuracy']:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
```

---

## Key Insights

### Meta-Learning Best Practices

1. **Inner steps**: Few steps for fast adaptation
2. **Outer LR**: Smaller than inner LR
3. **Support size**: Sufficient for adaptation
4. **Task diversity**: Varied tasks for generalization

---

## Exercises

1. **Exercise 1**: Implement Reptile
2. **Exercise 2**: Add first-order MAML
3. **Exercise 3**: Design task-aware aggregation
4. **Exercise 4**: Add ProtoNets

---

## References

1. Finn, C., et al. (2017). Model-agnostic meta-learning for fast adaptation. In *ICML*.
2. Fallah, A., et al. (2020). Personalized federated learning: A meta-learning approach. In *NeurIPS*.
3. Nichol, A., et al. (2018). On first-order meta-learning algorithms. *arXiv*.

---

*Copyright 2026 Olaf Yunus Laitinen Imanov and Contributors. Released under EUPL 1.2.*
