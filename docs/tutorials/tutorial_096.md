# Tutorial 096: FL Model Inversion Defense

---

## Metadata

| Property | Value |
|----------|-------|
| **Tutorial ID** | 096 |
| **Title** | FL Model Inversion Defense |
| **Category** | Security |
| **Difficulty** | Expert |
| **Duration** | 90 minutes |
| **Prerequisites** | Tutorial 001-095 |
| **Author** | Unbitrium Contributors |
| **Last Updated** | January 2026 |

---

## Learning Objectives

By completing this tutorial, you will be able to:

1. **Understand** model inversion attacks
2. **Implement** defense mechanisms
3. **Design** privacy-preserving inference
4. **Analyze** defense effectiveness
5. **Deploy** robust FL systems

---

## Background and Theory

### Model Inversion Defenses

```
Defense Strategies:
├── Differential Privacy
│   ├── Gradient noise
│   ├── Output perturbation
│   └── Label smoothing
├── Model Regularization
│   ├── Dropout
│   ├── Early stopping
│   └── Weight decay
├── Access Control
│   ├── Query limits
│   ├── Rate limiting
│   └── Authentication
└── Output Modification
    ├── Confidence masking
    ├── Thresholding
    └── Ensemble voting
```

---

## Implementation Code

```python
#!/usr/bin/env python3
"""
Tutorial 096: FL Model Inversion Defense

This module implements defenses against model
inversion attacks in federated learning.

Copyright 2026 Olaf Yunus Laitinen Imanov and Contributors
Released under EUPL 1.2
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass
from typing import Dict, List
import copy
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DefenseConfig:
    """Defense configuration."""
    
    num_rounds: int = 30
    num_clients: int = 10
    clients_per_round: int = 5
    
    input_dim: int = 32
    hidden_dim: int = 64
    num_classes: int = 10
    
    learning_rate: float = 0.01
    batch_size: int = 32
    local_epochs: int = 3
    
    # Defense params
    dp_epsilon: float = 1.0
    dp_delta: float = 1e-5
    clip_norm: float = 1.0
    
    confidence_threshold: float = 0.9
    label_smoothing: float = 0.1
    
    seed: int = 42


class GradientDefense:
    """Defense via gradient perturbation."""
    
    def __init__(self, config: DefenseConfig):
        self.config = config
    
    def clip_gradients(self, model: nn.Module) -> float:
        """Clip gradient norms."""
        total_norm = 0.0
        for param in model.parameters():
            if param.grad is not None:
                total_norm += param.grad.data.norm() ** 2
        total_norm = total_norm ** 0.5
        
        clip_factor = min(1.0, self.config.clip_norm / (total_norm + 1e-8))
        
        for param in model.parameters():
            if param.grad is not None:
                param.grad.data.mul_(clip_factor)
        
        return total_norm.item()
    
    def add_noise(self, model: nn.Module, num_samples: int) -> None:
        """Add calibrated DP noise."""
        noise_scale = self.config.clip_norm * np.sqrt(
            2 * np.log(1.25 / self.config.dp_delta)
        ) / (self.config.dp_epsilon * num_samples)
        
        for param in model.parameters():
            if param.grad is not None:
                noise = torch.randn_like(param.grad) * noise_scale
                param.grad.data.add_(noise)


class OutputDefense:
    """Defense via output modification."""
    
    def __init__(self, config: DefenseConfig):
        self.config = config
    
    def mask_confidence(self, logits: torch.Tensor) -> torch.Tensor:
        """Mask high-confidence predictions."""
        probs = F.softmax(logits, dim=-1)
        max_probs, _ = probs.max(dim=-1, keepdim=True)
        
        # Apply threshold
        mask = (max_probs < self.config.confidence_threshold).float()
        noise = torch.randn_like(probs) * 0.1
        
        return probs * (1 - mask) + (probs + noise) * mask
    
    def add_output_noise(self, logits: torch.Tensor, scale: float = 0.1) -> torch.Tensor:
        """Add noise to outputs."""
        noise = torch.randn_like(logits) * scale
        return logits + noise


class LabelSmoothing(nn.Module):
    """Label smoothing for training."""
    
    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        self.smoothing = smoothing
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        n_classes = logits.size(-1)
        
        # Create smoothed labels
        targets_one_hot = F.one_hot(targets, n_classes).float()
        smooth_targets = targets_one_hot * (1 - self.smoothing) + self.smoothing / n_classes
        
        log_probs = F.log_softmax(logits, dim=-1)
        loss = -(smooth_targets * log_probs).sum(dim=-1).mean()
        
        return loss


class DefenseDataset(Dataset):
    def __init__(self, n: int = 200, dim: int = 32, classes: int = 10, seed: int = 0):
        np.random.seed(seed)
        self.x = torch.randn(n, dim, dtype=torch.float32)
        self.y = torch.randint(0, classes, (n,), dtype=torch.long)
        for i in range(n):
            self.x[i, self.y[i].item() % dim] += 2.0
    
    def __len__(self): return len(self.y)
    def __getitem__(self, idx): return self.x[idx], self.y[idx]


class DefenseModel(nn.Module):
    def __init__(self, config: DefenseConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(config.hidden_dim, config.num_classes)
        )
    
    def forward(self, x): return self.net(x)


class DefenseClient:
    """Client with defense mechanisms."""
    
    def __init__(self, client_id: int, dataset: DefenseDataset, config: DefenseConfig):
        self.client_id = client_id
        self.dataset = dataset
        self.config = config
        
        self.grad_defense = GradientDefense(config)
        self.label_smoothing = LabelSmoothing(config.label_smoothing)
    
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
                
                output = local(x)
                loss = self.label_smoothing(output, y)
                loss.backward()
                
                # Apply gradient defense
                self.grad_defense.clip_gradients(local)
                self.grad_defense.add_noise(local, len(self.dataset))
                
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
        
        return {
            "state_dict": {k: v.cpu() for k, v in local.state_dict().items()},
            "num_samples": len(self.dataset),
            "avg_loss": total_loss / num_batches
        }


class DefenseServer:
    """Server with defense mechanisms."""
    
    def __init__(
        self,
        model: nn.Module,
        clients: List[DefenseClient],
        test_data: DefenseDataset,
        config: DefenseConfig
    ):
        self.model = model
        self.clients = clients
        self.test_data = test_data
        self.config = config
        
        self.output_defense = OutputDefense(config)
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
        correct, total = 0, 0
        
        with torch.no_grad():
            for x, y in loader:
                logits = self.model(x)
                # Apply output defense
                probs = self.output_defense.mask_confidence(logits)
                pred = probs.argmax(dim=1)
                correct += (pred == y).sum().item()
                total += len(y)
        
        return {"accuracy": correct / total}
    
    def train(self) -> List[Dict]:
        logger.info(f"Starting FL with defenses (ε={self.config.dp_epsilon})")
        
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
                logger.info(f"Round {round_num + 1}: acc={metrics['accuracy']:.4f}")
        
        return self.history


def main():
    print("=" * 60)
    print("Tutorial 096: FL Model Inversion Defense")
    print("=" * 60)
    
    config = DefenseConfig()
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    
    clients = [DefenseClient(i, DefenseDataset(seed=config.seed + i), config) for i in range(config.num_clients)]
    test_data = DefenseDataset(seed=999)
    model = DefenseModel(config)
    
    server = DefenseServer(model, clients, test_data, config)
    history = server.train()
    
    print("\n" + "=" * 60)
    print("Defense Training Complete")
    print(f"Final accuracy: {history[-1]['accuracy']:.4f}")
    print(f"DP epsilon: {config.dp_epsilon}")
    print(f"Label smoothing: {config.label_smoothing}")
    print("=" * 60)


if __name__ == "__main__":
    main()
```

---

## Key Insights

### Defense Best Practices

1. **Layered defense**: Combine multiple techniques
2. **Privacy budget**: Track epsilon usage
3. **Utility trade-off**: Balance privacy and accuracy
4. **Regular audits**: Test against known attacks

---

## Exercises

1. **Exercise 1**: Add query auditing
2. **Exercise 2**: Implement ensemble defense
3. **Exercise 3**: Design adaptive defense
4. **Exercise 4**: Measure attack resistance

---

## References

1. Fredrikson, M., et al. (2015). Model inversion attacks. In *CCS*.
2. Abadi, M., et al. (2016). Deep learning with differential privacy. In *CCS*.
3. Jia, J., et al. (2019). MemGuard: Defending against black-box membership inference attacks. In *CCS*.

---

*Copyright 2026 Olaf Yunus Laitinen Imanov and Contributors. Released under EUPL 1.2.*
