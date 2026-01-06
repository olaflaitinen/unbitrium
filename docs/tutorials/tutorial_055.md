# Tutorial 055: FL Knowledge Distillation

---

## Metadata

| Property | Value |
|----------|-------|
| **Tutorial ID** | 055 |
| **Title** | Federated Learning Knowledge Distillation |
| **Category** | Advanced Techniques |
| **Difficulty** | Advanced |
| **Duration** | 90 minutes |
| **Prerequisites** | Tutorial 001-054 |
| **Author** | Unbitrium Contributors |
| **Last Updated** | January 2026 |

---

## Learning Objectives

By completing this tutorial, you will be able to:

1. **Understand** knowledge distillation in FL
2. **Implement** federated distillation
3. **Design** teacher-student FL systems
4. **Analyze** distillation efficiency
5. **Deploy** communication-efficient FL

---

## Prerequisites

Before starting this tutorial, ensure you have:

- Completed tutorials 001-054
- Understanding of FL fundamentals
- Knowledge of knowledge distillation
- Familiarity with model compression

---

## Background and Theory

### Knowledge Distillation in FL

```
FL Distillation:
├── Standard Distillation
│   ├── Teacher-student
│   ├── Soft labels
│   └── Temperature scaling
├── Federated Distillation
│   ├── FedMD (logit sharing)
│   ├── FedDF (data-free)
│   └── Ensemble distillation
├── Communication Efficiency
│   ├── Share logits vs gradients
│   ├── Compressed representations
│   └── Selective sharing
└── Heterogeneous Models
    ├── Different architectures
    ├── Client-specific models
    └── Personalization
```

---

## Implementation Code

```python
#!/usr/bin/env python3
"""
Tutorial 055: FL Knowledge Distillation

This module implements knowledge distillation for
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
from typing import Dict, List, Optional
import copy
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class KDConfig:
    """Knowledge distillation configuration."""
    
    num_rounds: int = 50
    num_clients: int = 10
    clients_per_round: int = 5
    
    input_dim: int = 32
    teacher_hidden: int = 128
    student_hidden: int = 32
    num_classes: int = 10
    
    learning_rate: float = 0.01
    batch_size: int = 32
    local_epochs: int = 3
    
    temperature: float = 3.0
    alpha: float = 0.5  # Weight for distillation loss
    
    seed: int = 42


class TeacherModel(nn.Module):
    """Larger teacher model."""
    
    def __init__(self, config: KDConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.input_dim, config.teacher_hidden),
            nn.ReLU(),
            nn.Linear(config.teacher_hidden, config.teacher_hidden // 2),
            nn.ReLU(),
            nn.Linear(config.teacher_hidden // 2, config.num_classes)
        )
    
    def forward(self, x): return self.net(x)


class StudentModel(nn.Module):
    """Smaller student model."""
    
    def __init__(self, config: KDConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.input_dim, config.student_hidden),
            nn.ReLU(),
            nn.Linear(config.student_hidden, config.num_classes)
        )
    
    def forward(self, x): return self.net(x)


class KDDataset(Dataset):
    def __init__(self, n: int = 200, dim: int = 32, classes: int = 10, seed: int = 0):
        np.random.seed(seed)
        self.x = torch.randn(n, dim, dtype=torch.float32)
        self.y = torch.randint(0, classes, (n,), dtype=torch.long)
        for i in range(n):
            self.x[i, self.y[i].item() % dim] += 2.0
    
    def __len__(self): return len(self.y)
    def __getitem__(self, idx): return self.x[idx], self.y[idx]


def distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    temperature: float,
    alpha: float
) -> torch.Tensor:
    """Compute distillation loss."""
    soft_teacher = F.softmax(teacher_logits / temperature, dim=1)
    soft_student = F.log_softmax(student_logits / temperature, dim=1)
    
    distill_loss = F.kl_div(soft_student, soft_teacher, reduction='batchmean') * (temperature ** 2)
    hard_loss = F.cross_entropy(student_logits, labels)
    
    return alpha * distill_loss + (1 - alpha) * hard_loss


class KDClient:
    """Client with knowledge distillation."""
    
    def __init__(
        self,
        client_id: int,
        dataset: KDDataset,
        config: KDConfig
    ):
        self.client_id = client_id
        self.dataset = dataset
        self.config = config
        
        # Local teacher
        self.teacher = TeacherModel(config)
    
    def train_teacher(self, global_teacher: nn.Module) -> Dict:
        """Train local teacher."""
        local = copy.deepcopy(global_teacher)
        optimizer = torch.optim.Adam(local.parameters(), lr=self.config.learning_rate)
        loader = DataLoader(self.dataset, batch_size=self.config.batch_size, shuffle=True)
        
        local.train()
        for _ in range(self.config.local_epochs):
            for x, y in loader:
                optimizer.zero_grad()
                loss = F.cross_entropy(local(x), y)
                loss.backward()
                optimizer.step()
        
        self.teacher.load_state_dict(local.state_dict())
        
        return {
            "state_dict": {k: v.cpu() for k, v in local.state_dict().items()},
            "num_samples": len(self.dataset)
        }
    
    def train_student(self, student: nn.Module) -> Dict:
        """Train student with distillation from local teacher."""
        local = copy.deepcopy(student)
        optimizer = torch.optim.Adam(local.parameters(), lr=self.config.learning_rate)
        loader = DataLoader(self.dataset, batch_size=self.config.batch_size, shuffle=True)
        
        local.train()
        self.teacher.eval()
        
        total_loss = 0.0
        num_batches = 0
        
        for _ in range(self.config.local_epochs):
            for x, y in loader:
                optimizer.zero_grad()
                
                student_out = local(x)
                
                with torch.no_grad():
                    teacher_out = self.teacher(x)
                
                loss = distillation_loss(
                    student_out, teacher_out, y,
                    self.config.temperature, self.config.alpha
                )
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
        
        return {
            "state_dict": {k: v.cpu() for k, v in local.state_dict().items()},
            "num_samples": len(self.dataset),
            "avg_loss": total_loss / num_batches
        }


class KDServer:
    """Server for FL with knowledge distillation."""
    
    def __init__(
        self,
        clients: List[KDClient],
        test_data: KDDataset,
        config: KDConfig
    ):
        self.clients = clients
        self.test_data = test_data
        self.config = config
        
        self.teacher = TeacherModel(config)
        self.student = StudentModel(config)
        self.history: List[Dict] = []
    
    def aggregate(self, updates: List[Dict], model: nn.Module) -> None:
        total = sum(u["num_samples"] for u in updates)
        new_state = {}
        for key in updates[0]["state_dict"]:
            new_state[key] = sum(
                (u["num_samples"] / total) * u["state_dict"][key].float()
                for u in updates
            )
        model.load_state_dict(new_state)
    
    def evaluate(self, model: nn.Module) -> Dict[str, float]:
        model.eval()
        loader = DataLoader(self.test_data, batch_size=64)
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in loader:
                pred = model(x).argmax(dim=1)
                correct += (pred == y).sum().item()
                total += len(y)
        return {"accuracy": correct / total}
    
    def train(self) -> List[Dict]:
        logger.info("Starting FL with knowledge distillation")
        
        # Phase 1: Train teacher
        logger.info("Phase 1: Training teacher...")
        for round_num in range(self.config.num_rounds // 2):
            n = min(self.config.clients_per_round, len(self.clients))
            indices = np.random.choice(len(self.clients), n, replace=False)
            selected = [self.clients[i] for i in indices]
            
            updates = [c.train_teacher(self.teacher) for c in selected]
            self.aggregate(updates, self.teacher)
        
        teacher_metrics = self.evaluate(self.teacher)
        logger.info(f"Teacher accuracy: {teacher_metrics['accuracy']:.4f}")
        
        # Phase 2: Train student with distillation
        logger.info("Phase 2: Training student with distillation...")
        for round_num in range(self.config.num_rounds // 2):
            n = min(self.config.clients_per_round, len(self.clients))
            indices = np.random.choice(len(self.clients), n, replace=False)
            selected = [self.clients[i] for i in indices]
            
            updates = [c.train_student(self.student) for c in selected]
            self.aggregate(updates, self.student)
            
            metrics = self.evaluate(self.student)
            
            record = {"round": round_num, **metrics}
            self.history.append(record)
        
        return self.history


def main():
    print("=" * 60)
    print("Tutorial 055: FL Knowledge Distillation")
    print("=" * 60)
    
    config = KDConfig()
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    
    clients = [KDClient(i, KDDataset(seed=config.seed + i), config) for i in range(config.num_clients)]
    test_data = KDDataset(seed=999)
    
    server = KDServer(clients, test_data, config)
    history = server.train()
    
    # Compare sizes
    teacher_params = sum(p.numel() for p in server.teacher.parameters())
    student_params = sum(p.numel() for p in server.student.parameters())
    
    print("\n" + "=" * 60)
    print("Knowledge Distillation Complete")
    print(f"Teacher params: {teacher_params:,}")
    print(f"Student params: {student_params:,}")
    print(f"Compression: {teacher_params / student_params:.1f}x")
    print(f"Student accuracy: {history[-1]['accuracy']:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
```

---

## Key Insights

### Distillation Best Practices

1. **Temperature**: Higher for softer labels
2. **Alpha balance**: Trade-off distill vs hard loss
3. **Teacher quality**: Better teacher = better student
4. **Communication**: Logits cheaper than gradients

---

## Exercises

1. **Exercise 1**: Implement FedMD
2. **Exercise 2**: Add data-free distillation
3. **Exercise 3**: Design ensemble distillation
4. **Exercise 4**: Add heterogeneous models

---

## References

1. Hinton, G., et al. (2015). Distilling the knowledge in a neural network. *arXiv*.
2. Li, D., & Wang, J. (2019). FedMD: Heterogeneous FL via model distillation. *arXiv*.
3. Lin, Y., et al. (2020). Ensemble distillation for robust model fusion in FL. In *NeurIPS*.

---

*Copyright 2026 Olaf Yunus Laitinen Imanov and Contributors. Released under EUPL 1.2.*
