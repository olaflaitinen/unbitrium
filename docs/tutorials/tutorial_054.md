# Tutorial 054: FL Multi-Task Learning

---

## Metadata

| Property | Value |
|----------|-------|
| **Tutorial ID** | 054 |
| **Title** | Federated Learning Multi-Task Learning |
| **Category** | Advanced Techniques |
| **Difficulty** | Advanced |
| **Duration** | 90 minutes |
| **Prerequisites** | Tutorial 001-053 |
| **Author** | Unbitrium Contributors |
| **Last Updated** | January 2026 |

---

## Learning Objectives

By completing this tutorial, you will be able to:

1. **Understand** multi-task FL
2. **Implement** shared representation learning
3. **Design** task-specific heads
4. **Analyze** task relationships
5. **Deploy** multi-task FL systems

---

## Prerequisites

Before starting this tutorial, ensure you have:

- Completed tutorials 001-053
- Understanding of FL fundamentals
- Knowledge of multi-task learning
- Familiarity with shared representations

---

## Background and Theory

### Multi-Task FL Architecture

```
Multi-Task FL:
├── Shared Components
│   ├── Backbone network
│   ├── Feature extractor
│   └── Embedding layers
├── Task-Specific
│   ├── Classification head
│   ├── Regression head
│   └── Segmentation head
├── Learning Strategies
│   ├── Hard parameter sharing
│   ├── Soft parameter sharing
│   └── Cross-stitch networks
└── Aggregation
    ├── Per-task averaging
    ├── Weighted by task
    └── Gradient balancing
```

---

## Implementation Code

```python
#!/usr/bin/env python3
"""
Tutorial 054: FL Multi-Task Learning

This module implements multi-task learning for
federated learning scenarios.

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
class MTLConfig:
    """Multi-task learning configuration."""
    
    num_rounds: int = 50
    num_clients: int = 10
    clients_per_round: int = 5
    
    input_dim: int = 32
    hidden_dim: int = 64
    num_classes_task1: int = 10
    num_classes_task2: int = 5
    
    learning_rate: float = 0.01
    batch_size: int = 32
    local_epochs: int = 3
    
    task1_weight: float = 0.5
    task2_weight: float = 0.5
    
    seed: int = 42


class MultiTaskModel(nn.Module):
    """Model with shared backbone and multiple heads."""
    
    def __init__(self, config: MTLConfig):
        super().__init__()
        
        # Shared backbone
        self.backbone = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU()
        )
        
        # Task-specific heads
        self.head1 = nn.Linear(config.hidden_dim // 2, config.num_classes_task1)
        self.head2 = nn.Linear(config.hidden_dim // 2, config.num_classes_task2)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(x)
        out1 = self.head1(features)
        out2 = self.head2(features)
        return out1, out2
    
    def get_backbone_params(self):
        return self.backbone.parameters()
    
    def get_head_params(self, task: int):
        if task == 1:
            return self.head1.parameters()
        return self.head2.parameters()


class MultiTaskDataset(Dataset):
    """Dataset with multiple task labels."""
    
    def __init__(
        self,
        n: int = 200,
        dim: int = 32,
        classes1: int = 10,
        classes2: int = 5,
        seed: int = 0
    ):
        np.random.seed(seed)
        self.x = torch.randn(n, dim, dtype=torch.float32)
        self.y1 = torch.randint(0, classes1, (n,), dtype=torch.long)
        self.y2 = torch.randint(0, classes2, (n,), dtype=torch.long)
        
        for i in range(n):
            self.x[i, self.y1[i].item() % dim] += 2.0
            self.x[i, (self.y2[i].item() + 5) % dim] += 1.0
    
    def __len__(self): return len(self.y1)
    
    def __getitem__(self, idx):
        return self.x[idx], self.y1[idx], self.y2[idx]


class MTLClient:
    """Multi-task FL client."""
    
    def __init__(
        self,
        client_id: int,
        dataset: MultiTaskDataset,
        config: MTLConfig
    ):
        self.client_id = client_id
        self.dataset = dataset
        self.config = config
    
    def train(self, model: nn.Module) -> Dict:
        local = copy.deepcopy(model)
        optimizer = torch.optim.Adam(local.parameters(), lr=self.config.learning_rate)
        loader = DataLoader(self.dataset, batch_size=self.config.batch_size, shuffle=True)
        
        local.train()
        total_loss, num_batches = 0.0, 0
        task1_loss_sum, task2_loss_sum = 0.0, 0.0
        
        for _ in range(self.config.local_epochs):
            for x, y1, y2 in loader:
                optimizer.zero_grad()
                
                out1, out2 = local(x)
                
                loss1 = F.cross_entropy(out1, y1)
                loss2 = F.cross_entropy(out2, y2)
                
                # Weighted multi-task loss
                loss = self.config.task1_weight * loss1 + self.config.task2_weight * loss2
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                task1_loss_sum += loss1.item()
                task2_loss_sum += loss2.item()
                num_batches += 1
        
        return {
            "state_dict": {k: v.cpu() for k, v in local.state_dict().items()},
            "num_samples": len(self.dataset),
            "avg_loss": total_loss / num_batches,
            "task1_loss": task1_loss_sum / num_batches,
            "task2_loss": task2_loss_sum / num_batches
        }


class MTLServer:
    """Multi-task FL server."""
    
    def __init__(
        self,
        model: nn.Module,
        clients: List[MTLClient],
        test_data: MultiTaskDataset,
        config: MTLConfig
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
        
        correct1, correct2, total = 0, 0, 0
        
        with torch.no_grad():
            for x, y1, y2 in loader:
                out1, out2 = self.model(x)
                
                pred1 = out1.argmax(dim=1)
                pred2 = out2.argmax(dim=1)
                
                correct1 += (pred1 == y1).sum().item()
                correct2 += (pred2 == y2).sum().item()
                total += len(y1)
        
        return {
            "task1_accuracy": correct1 / total,
            "task2_accuracy": correct2 / total
        }
    
    def train(self) -> List[Dict]:
        logger.info("Starting multi-task FL")
        
        for round_num in range(self.config.num_rounds):
            n = min(self.config.clients_per_round, len(self.clients))
            indices = np.random.choice(len(self.clients), n, replace=False)
            selected = [self.clients[i] for i in indices]
            
            updates = [c.train(self.model) for c in selected]
            self.aggregate(updates)
            
            metrics = self.evaluate()
            
            record = {"round": round_num, **metrics}
            self.history.append(record)
            
            if (round_num + 1) % 10 == 0:
                logger.info(
                    f"Round {round_num + 1}: "
                    f"task1={metrics['task1_accuracy']:.4f}, "
                    f"task2={metrics['task2_accuracy']:.4f}"
                )
        
        return self.history


def main():
    print("=" * 60)
    print("Tutorial 054: FL Multi-Task Learning")
    print("=" * 60)
    
    config = MTLConfig()
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    
    clients = [
        MTLClient(i, MultiTaskDataset(seed=config.seed + i), config)
        for i in range(config.num_clients)
    ]
    test_data = MultiTaskDataset(seed=999)
    model = MultiTaskModel(config)
    
    server = MTLServer(model, clients, test_data, config)
    history = server.train()
    
    print("\n" + "=" * 60)
    print("Multi-Task FL Complete")
    print(f"Task 1 accuracy: {history[-1]['task1_accuracy']:.4f}")
    print(f"Task 2 accuracy: {history[-1]['task2_accuracy']:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
```

---

## Key Insights

### Multi-Task Best Practices

1. **Share wisely**: Common features in backbone
2. **Balance tasks**: Weight losses appropriately
3. **Task relationships**: Group related tasks
4. **Gradient surgery**: Handle conflicting gradients

---

## Exercises

1. **Exercise 1**: Add gradient balancing
2. **Exercise 2**: Implement soft parameter sharing
3. **Exercise 3**: Design task-specific aggregation
4. **Exercise 4**: Add uncertainty weighting

---

## References

1. Caruana, R. (1997). Multitask learning. *Machine Learning*.
2. Chen, Z., et al. (2018). GradNorm: Gradient normalization for adaptive loss balancing. In *ICML*.
3. Smith, V., et al. (2017). Federated multi-task learning. In *NeurIPS*.

---

*Copyright 2026 Olaf Yunus Laitinen Imanov and Contributors. Released under EUPL 1.2.*
