# Tutorial 053: FL Transfer Learning

---

## Metadata

| Property | Value |
|----------|-------|
| **Tutorial ID** | 053 |
| **Title** | Federated Learning Transfer Learning |
| **Category** | Advanced Techniques |
| **Difficulty** | Advanced |
| **Duration** | 90 minutes |
| **Prerequisites** | Tutorial 001-052 |
| **Author** | Unbitrium Contributors |
| **Last Updated** | January 2026 |

---

## Learning Objectives

By completing this tutorial, you will be able to:

1. **Understand** transfer learning in FL
2. **Implement** federated pre-training
3. **Design** domain adaptation strategies
4. **Analyze** knowledge transfer
5. **Deploy** transfer-based FL systems

---

## Prerequisites

Before starting this tutorial, ensure you have:

- Completed tutorials 001-052
- Understanding of FL fundamentals
- Knowledge of transfer learning
- Familiarity with pre-trained models

---

## Background and Theory

### Transfer Learning in FL

```
FL Transfer Learning:
├── Pre-training
│   ├── Federated pre-training
│   ├── Centralized pre-training
│   └── Foundation models
├── Fine-tuning
│   ├── Full fine-tuning
│   ├── Head-only
│   └── Adapter tuning
├── Domain Adaptation
│   ├── Source-target alignment
│   ├── Feature matching
│   └── Distribution alignment
└── Knowledge Transfer
    ├── Cross-silo transfer
    ├── Cross-domain transfer
    └── Multi-task transfer
```

---

## Implementation Code

```python
#!/usr/bin/env python3
"""
Tutorial 053: FL Transfer Learning

This module implements transfer learning for
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
from typing import Dict, List, Optional, Tuple
from enum import Enum
import copy
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TransferMode(Enum):
    FULL = "full"
    HEAD_ONLY = "head_only"
    FREEZE_BACKBONE = "freeze_backbone"


@dataclass
class TransferConfig:
    """Transfer learning configuration."""
    
    pre_train_rounds: int = 20
    fine_tune_rounds: int = 30
    num_clients: int = 10
    clients_per_round: int = 5
    
    input_dim: int = 32
    hidden_dim: int = 64
    num_classes: int = 10
    target_classes: int = 5
    
    learning_rate: float = 0.01
    fine_tune_lr: float = 0.001
    batch_size: int = 32
    local_epochs: int = 3
    
    transfer_mode: TransferMode = TransferMode.FREEZE_BACKBONE
    
    seed: int = 42


class TransferModel(nn.Module):
    """Model with backbone and head."""
    
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int):
        super().__init__()
        
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )
        
        self.head = nn.Linear(hidden_dim // 2, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.head(features)
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)
    
    def freeze_backbone(self) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = False
    
    def unfreeze_backbone(self) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = True
    
    def replace_head(self, num_classes: int) -> None:
        in_features = self.head.in_features
        self.head = nn.Linear(in_features, num_classes)


class SourceDataset(Dataset):
    """Source domain dataset for pre-training."""
    
    def __init__(self, n: int = 500, dim: int = 32, classes: int = 10, seed: int = 0):
        np.random.seed(seed)
        self.x = torch.randn(n, dim, dtype=torch.float32)
        self.y = torch.randint(0, classes, (n,), dtype=torch.long)
        for i in range(n):
            self.x[i, self.y[i].item() % dim] += 2.0
    
    def __len__(self): return len(self.y)
    def __getitem__(self, idx): return self.x[idx], self.y[idx]


class TargetDataset(Dataset):
    """Target domain dataset for fine-tuning."""
    
    def __init__(self, n: int = 100, dim: int = 32, classes: int = 5, seed: int = 0, shift: float = 0.5):
        np.random.seed(seed)
        self.x = torch.randn(n, dim, dtype=torch.float32) + shift
        self.y = torch.randint(0, classes, (n,), dtype=torch.long)
        for i in range(n):
            self.x[i, self.y[i].item() % dim] += 1.5
    
    def __len__(self): return len(self.y)
    def __getitem__(self, idx): return self.x[idx], self.y[idx]


class TransferClient:
    def __init__(self, client_id: int, dataset: Dataset, config: TransferConfig, is_source: bool = True):
        self.client_id = client_id
        self.dataset = dataset
        self.config = config
        self.is_source = is_source
    
    def train(self, model: nn.Module, lr: float) -> Dict:
        local = copy.deepcopy(model)
        
        # Get trainable params
        params = [p for p in local.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(params, lr=lr)
        loader = DataLoader(self.dataset, batch_size=self.config.batch_size, shuffle=True)
        
        local.train()
        total_loss, num_batches = 0.0, 0
        
        for _ in range(self.config.local_epochs):
            for x, y in loader:
                optimizer.zero_grad()
                loss = F.cross_entropy(local(x), y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                num_batches += 1
        
        return {
            "state_dict": {k: v.cpu() for k, v in local.state_dict().items()},
            "num_samples": len(self.dataset),
            "avg_loss": total_loss / num_batches
        }


class TransferServer:
    """Server for FL with transfer learning."""
    
    def __init__(
        self,
        source_clients: List[TransferClient],
        target_clients: List[TransferClient],
        source_test: SourceDataset,
        target_test: TargetDataset,
        config: TransferConfig
    ):
        self.source_clients = source_clients
        self.target_clients = target_clients
        self.source_test = source_test
        self.target_test = target_test
        self.config = config
        
        self.model = TransferModel(config.input_dim, config.hidden_dim, config.num_classes)
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
    
    def evaluate(self, test_data: Dataset) -> Dict[str, float]:
        self.model.eval()
        loader = DataLoader(test_data, batch_size=64)
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in loader:
                pred = self.model(x).argmax(dim=1)
                correct += (pred == y).sum().item()
                total += len(y)
        return {"accuracy": correct / total}
    
    def pre_train(self) -> None:
        logger.info("Pre-training on source domain...")
        
        for round_num in range(self.config.pre_train_rounds):
            n = min(self.config.clients_per_round, len(self.source_clients))
            indices = np.random.choice(len(self.source_clients), n, replace=False)
            selected = [self.source_clients[i] for i in indices]
            
            updates = [c.train(self.model, self.config.learning_rate) for c in selected]
            self.aggregate(updates)
            
            if (round_num + 1) % 10 == 0:
                metrics = self.evaluate(self.source_test)
                logger.info(f"Pre-train round {round_num + 1}: acc={metrics['accuracy']:.4f}")
    
    def fine_tune(self) -> None:
        logger.info("Fine-tuning on target domain...")
        
        # Apply transfer mode
        if self.config.transfer_mode in [TransferMode.HEAD_ONLY, TransferMode.FREEZE_BACKBONE]:
            self.model.freeze_backbone()
        
        # Replace head for target task
        self.model.replace_head(self.config.target_classes)
        
        for round_num in range(self.config.fine_tune_rounds):
            n = min(self.config.clients_per_round, len(self.target_clients))
            indices = np.random.choice(len(self.target_clients), n, replace=False)
            selected = [self.target_clients[i] for i in indices]
            
            updates = [c.train(self.model, self.config.fine_tune_lr) for c in selected]
            self.aggregate(updates)
            
            if (round_num + 1) % 10 == 0:
                metrics = self.evaluate(self.target_test)
                logger.info(f"Fine-tune round {round_num + 1}: acc={metrics['accuracy']:.4f}")
                self.history.append({"round": round_num, "phase": "fine_tune", **metrics})
    
    def train(self) -> List[Dict]:
        self.pre_train()
        self.fine_tune()
        return self.history


def main():
    print("=" * 60)
    print("Tutorial 053: FL Transfer Learning")
    print("=" * 60)
    
    config = TransferConfig()
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    
    # Source clients
    source_clients = [TransferClient(i, SourceDataset(seed=config.seed + i), config, is_source=True) for i in range(config.num_clients)]
    
    # Target clients
    target_clients = [TransferClient(i + 100, TargetDataset(seed=config.seed + i + 100), config, is_source=False) for i in range(config.num_clients)]
    
    source_test = SourceDataset(seed=999)
    target_test = TargetDataset(seed=998)
    
    server = TransferServer(source_clients, target_clients, source_test, target_test, config)
    history = server.train()
    
    print("\n" + "=" * 60)
    print("Transfer Learning Complete")
    print(f"Final target accuracy: {history[-1]['accuracy']:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
```

---

## Key Insights

### Transfer Best Practices

1. **Pre-train broadly**: Large source domain
2. **Freeze smart**: Only fine-tune needed layers
3. **Lower LR**: Smaller learning rate for fine-tuning
4. **Evaluate both**: Track source and target performance

---

## Exercises

1. **Exercise 1**: Implement adapter tuning
2. **Exercise 2**: Add domain adversarial training
3. **Exercise 3**: Design multi-source transfer
4. **Exercise 4**: Add progressive unfreezing

---

## References

1. Chen, H.Y., & Chao, W.L. (2021). FedBE: Making Bayesian model ensemble applicable to FL. In *ICLR*.
2. Yu, T., et al. (2020). Federated transfer learning. In *FL Book*.
3. Peng, X., et al. (2019). Moment matching for multi-source domain adaptation. In *ICCV*.

---

*Copyright 2026 Olaf Yunus Laitinen Imanov and Contributors. Released under EUPL 1.2.*
