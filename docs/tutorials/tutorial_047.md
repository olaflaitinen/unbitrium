# Tutorial 047: FL Aggregation Strategies

---

## Metadata

| Property | Value |
|----------|-------|
| **Tutorial ID** | 047 |
| **Title** | Federated Learning Aggregation Strategies |
| **Category** | Core Algorithms |
| **Difficulty** | Intermediate |
| **Duration** | 90 minutes |
| **Prerequisites** | Tutorial 001-046 |
| **Author** | Unbitrium Contributors |
| **Last Updated** | January 2026 |

---

## Learning Objectives

By completing this tutorial, you will be able to:

1. **Understand** aggregation algorithms in FL
2. **Implement** various aggregation methods
3. **Compare** FedAvg, FedProx, FedOpt
4. **Analyze** aggregation impact
5. **Deploy** optimal aggregation strategies

---

## Prerequisites

Before starting this tutorial, ensure you have:

- Completed tutorials 001-046
- Understanding of FL fundamentals
- Knowledge of optimization
- Familiarity with gradient methods

---

## Background and Theory

### Aggregation Methods

```
Aggregation Strategies:
├── Averaging Methods
│   ├── FedAvg (weighted average)
│   ├── FedProx (proximal term)
│   └── Simple average
├── Momentum Methods
│   ├── FedMom
│   ├── Server momentum
│   └── Adaptive momentum
├── Adaptive Methods
│   ├── FedAdam
│   ├── FedYogi
│   └── FedAdagrad
└── Robust Methods
    ├── Trimmed mean
    ├── Median
    └── Krum
```

---

## Implementation Code

```python
#!/usr/bin/env python3
"""
Tutorial 047: FL Aggregation Strategies

This module implements various aggregation strategies
for federated learning.

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


class AggregationMethod(Enum):
    FEDAVG = "fedavg"
    FEDPROX = "fedprox"
    FEDADAM = "fedadam"
    MEDIAN = "median"
    TRIMMED_MEAN = "trimmed_mean"


@dataclass
class AggConfig:
    """Aggregation configuration."""
    
    num_rounds: int = 50
    num_clients: int = 15
    clients_per_round: int = 8
    
    input_dim: int = 32
    hidden_dim: int = 64
    num_classes: int = 10
    
    learning_rate: float = 0.01
    batch_size: int = 32
    local_epochs: int = 3
    
    # Aggregation params
    method: AggregationMethod = AggregationMethod.FEDAVG
    prox_mu: float = 0.01
    server_lr: float = 1.0
    beta1: float = 0.9
    beta2: float = 0.99
    epsilon: float = 1e-8
    trim_ratio: float = 0.1
    
    seed: int = 42


class Aggregator:
    """Aggregation algorithms."""
    
    def __init__(self, config: AggConfig, model: nn.Module):
        self.config = config
        self.model = model
        
        # For FedAdam
        self.m = {k: torch.zeros_like(v) for k, v in model.state_dict().items()}
        self.v = {k: torch.zeros_like(v) for k, v in model.state_dict().items()}
        self.t = 0
    
    def aggregate(
        self,
        updates: List[Dict],
        global_state: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Aggregate based on method."""
        method = self.config.method
        
        if method == AggregationMethod.FEDAVG:
            return self._fedavg(updates)
        elif method == AggregationMethod.FEDPROX:
            return self._fedavg(updates)  # Prox term in client
        elif method == AggregationMethod.FEDADAM:
            return self._fedadam(updates, global_state)
        elif method == AggregationMethod.MEDIAN:
            return self._median(updates)
        elif method == AggregationMethod.TRIMMED_MEAN:
            return self._trimmed_mean(updates)
        
        return self._fedavg(updates)
    
    def _fedavg(self, updates: List[Dict]) -> Dict[str, torch.Tensor]:
        """FedAvg: weighted average by samples."""
        total = sum(u["num_samples"] for u in updates)
        new_state = {}
        
        for key in updates[0]["state_dict"]:
            new_state[key] = sum(
                (u["num_samples"] / total) * u["state_dict"][key].float()
                for u in updates
            )
        
        return new_state
    
    def _fedadam(
        self,
        updates: List[Dict],
        global_state: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """FedAdam: adaptive server optimizer."""
        self.t += 1
        
        # Compute pseudo-gradient
        avg_state = self._fedavg(updates)
        
        new_state = {}
        for key in avg_state:
            delta = avg_state[key] - global_state[key]
            
            # Update momentum
            self.m[key] = self.config.beta1 * self.m[key] + (1 - self.config.beta1) * delta
            self.v[key] = self.config.beta2 * self.v[key] + (1 - self.config.beta2) * (delta ** 2)
            
            # Bias correction
            m_hat = self.m[key] / (1 - self.config.beta1 ** self.t)
            v_hat = self.v[key] / (1 - self.config.beta2 ** self.t)
            
            # Update
            new_state[key] = global_state[key] + self.config.server_lr * m_hat / (torch.sqrt(v_hat) + self.config.epsilon)
        
        return new_state
    
    def _median(self, updates: List[Dict]) -> Dict[str, torch.Tensor]:
        """Coordinate-wise median."""
        new_state = {}
        
        for key in updates[0]["state_dict"]:
            stacked = torch.stack([u["state_dict"][key].float() for u in updates])
            new_state[key] = torch.median(stacked, dim=0)[0]
        
        return new_state
    
    def _trimmed_mean(self, updates: List[Dict]) -> Dict[str, torch.Tensor]:
        """Trimmed mean: remove outliers."""
        n = len(updates)
        trim = int(n * self.config.trim_ratio)
        
        new_state = {}
        
        for key in updates[0]["state_dict"]:
            stacked = torch.stack([u["state_dict"][key].float() for u in updates])
            
            # Sort and trim
            sorted_vals, _ = torch.sort(stacked, dim=0)
            trimmed = sorted_vals[trim:n - trim]
            
            new_state[key] = trimmed.mean(dim=0)
        
        return new_state


class AggDataset(Dataset):
    def __init__(self, n: int = 200, dim: int = 32, classes: int = 10, seed: int = 0):
        np.random.seed(seed)
        self.x = torch.randn(n, dim, dtype=torch.float32)
        self.y = torch.randint(0, classes, (n,), dtype=torch.long)
        for i in range(n):
            self.x[i, self.y[i].item() % dim] += 2.0
    
    def __len__(self): return len(self.y)
    def __getitem__(self, idx): return self.x[idx], self.y[idx]


class AggModel(nn.Module):
    def __init__(self, config: AggConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.num_classes)
        )
    
    def forward(self, x): return self.net(x)


class AggClient:
    def __init__(self, client_id: int, dataset: AggDataset, config: AggConfig):
        self.client_id = client_id
        self.dataset = dataset
        self.config = config
    
    def train(
        self,
        model: nn.Module,
        global_model: Optional[nn.Module] = None
    ) -> Dict:
        local = copy.deepcopy(model)
        optimizer = torch.optim.SGD(local.parameters(), lr=self.config.learning_rate)
        loader = DataLoader(self.dataset, batch_size=self.config.batch_size, shuffle=True)
        
        local.train()
        total_loss, num_batches = 0.0, 0
        
        for _ in range(self.config.local_epochs):
            for x, y in loader:
                optimizer.zero_grad()
                loss = F.cross_entropy(local(x), y)
                
                # FedProx term
                if self.config.method == AggregationMethod.FEDPROX and global_model:
                    prox = 0.0
                    for p_l, p_g in zip(local.parameters(), global_model.parameters()):
                        prox += ((p_l - p_g.detach()) ** 2).sum()
                    loss += self.config.prox_mu / 2 * prox
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
        
        return {
            "state_dict": {k: v.cpu() for k, v in local.state_dict().items()},
            "num_samples": len(self.dataset),
            "avg_loss": total_loss / num_batches
        }


class AggServer:
    def __init__(
        self,
        model: nn.Module,
        clients: List[AggClient],
        test_data: AggDataset,
        config: AggConfig
    ):
        self.model = model
        self.clients = clients
        self.test_data = test_data
        self.config = config
        
        self.aggregator = Aggregator(config, model)
        self.history: List[Dict] = []
    
    def evaluate(self) -> Dict[str, float]:
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
        logger.info(f"Starting FL with {self.config.method.value} aggregation")
        
        for round_num in range(self.config.num_rounds):
            n = min(self.config.clients_per_round, len(self.clients))
            indices = np.random.choice(len(self.clients), n, replace=False)
            selected = [self.clients[i] for i in indices]
            
            global_model = copy.deepcopy(self.model)
            updates = [c.train(self.model, global_model) for c in selected]
            
            global_state = self.model.state_dict()
            new_state = self.aggregator.aggregate(updates, global_state)
            self.model.load_state_dict(new_state)
            
            metrics = self.evaluate()
            
            record = {"round": round_num, **metrics}
            self.history.append(record)
            
            if (round_num + 1) % 10 == 0:
                logger.info(f"Round {round_num + 1}: acc={metrics['accuracy']:.4f}")
        
        return self.history


def main():
    print("=" * 60)
    print("Tutorial 047: FL Aggregation Strategies")
    print("=" * 60)
    
    config = AggConfig()
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    
    # Compare methods
    results = {}
    
    for method in AggregationMethod:
        config.method = method
        
        clients = [
            AggClient(i, AggDataset(seed=config.seed + i), config)
            for i in range(config.num_clients)
        ]
        test_data = AggDataset(seed=999)
        model = AggModel(config)
        
        server = AggServer(model, clients, test_data, config)
        history = server.train()
        
        results[method.value] = history[-1]["accuracy"]
    
    print("\n" + "=" * 60)
    print("Aggregation Comparison")
    for method, acc in results.items():
        print(f"  {method}: {acc:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
```

---

## Key Insights

### Aggregation Selection

1. **FedAvg**: Simple baseline
2. **FedProx**: Handles heterogeneity
3. **FedAdam**: Faster convergence
4. **Robust**: Defense against Byzantine

---

## Exercises

1. **Exercise 1**: Implement FedYogi
2. **Exercise 2**: Add server momentum
3. **Exercise 3**: Design adaptive selection
4. **Exercise 4**: Combine robust + adaptive

---

## References

1. McMahan, B., et al. (2017). Communication-efficient FL. In *AISTATS*.
2. Li, T., et al. (2020). Federated optimization in heterogeneous networks. In *MLSys*.
3. Reddi, S.J., et al. (2021). Adaptive federated optimization. In *ICLR*.

---

*Copyright 2026 Olaf Yunus Laitinen Imanov and Contributors. Released under EUPL 1.2.*
