# Tutorial 090: FL Regularization Techniques

---

## Metadata

| Property | Value |
|----------|-------|
| **Tutorial ID** | 090 |
| **Title** | FL Regularization Techniques |
| **Category** | Optimization |
| **Difficulty** | Intermediate |
| **Duration** | 90 minutes |
| **Prerequisites** | Tutorial 001-089 |
| **Author** | Unbitrium Contributors |
| **Last Updated** | January 2026 |

---

## Learning Objectives

By the end of this tutorial, you will be able to:

1. **Understand** regularization challenges in federated settings.
2. **Implement** L2, L1, and proximal regularization.
3. **Design** dropout and batch normalization for FL.
4. **Analyze** client drift mitigation techniques.
5. **Apply** regularization to prevent overfitting.
6. **Evaluate** regularization impact on generalization.
7. **Create** FL-specific regularization strategies.

---

## Prerequisites

- **Completed Tutorials**: 001-089
- **Knowledge**: Regularization, overfitting
- **Libraries**: PyTorch, NumPy

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple
from torch.utils.data import Dataset, DataLoader
import copy

print(f"PyTorch: {torch.__version__}")
```

---

## Background and Theory

### Regularization in FL

| Challenge | Description | Solution |
|-----------|-------------|----------|
| Client drift | Local models diverge | Proximal term |
| Overfitting | Small local data | Dropout, L2 |
| Heterogeneity | Different distributions | Personalization |
| Model divergence | Non-IID causes instability | Stronger regularization |

### Regularization Types

| Type | Effect | Formula |
|------|--------|---------|
| L2 (Weight decay) | Small weights | λ∥w∥² |
| L1 | Sparse weights | λ∥w∥₁ |
| Proximal (FedProx) | Stay near global | μ∥w - w_t∥² |
| Dropout | Random deactivation | p(keep) |
| Elastic Net | L1 + L2 combo | α∥w∥₁ + (1-α)∥w∥² |

### FedProx Algorithm

```mermaid
graph TB
    subgraph "Client k Training"
        GLOBAL[Global Model w_t]
        LOCAL[Local Model w_k]
        LOSS[Task Loss L(w_k)]
        PROX[Proximal Term μ/2 ∥w_k - w_t∥²]
        TOTAL[Total Loss]
    end

    GLOBAL --> PROX
    LOCAL --> PROX
    LOCAL --> LOSS
    LOSS --> TOTAL
    PROX --> TOTAL
```

---

## Implementation Code

### Part 1: Configuration and Regularizers

```python
#!/usr/bin/env python3
"""
Tutorial 090: FL Regularization Techniques

Comprehensive implementation of regularization techniques
for federated learning including proximal, L2, dropout.

Author: Unbitrium Contributors
License: EUPL-1.2
"""

from __future__ import annotations
import copy
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional
from enum import Enum
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


class RegularizationType(Enum):
    """Types of regularization."""
    NONE = "none"
    L2 = "l2"
    L1 = "l1"
    ELASTIC = "elastic"
    PROXIMAL = "proximal"
    COMBINED = "combined"


@dataclass
class RegConfig:
    """Configuration for regularization."""
    
    # General
    num_rounds: int = 100
    num_clients: int = 20
    clients_per_round: int = 10
    local_epochs: int = 5
    batch_size: int = 32
    learning_rate: float = 0.01
    seed: int = 42
    
    # Regularization
    reg_type: RegularizationType = RegularizationType.COMBINED
    l2_weight: float = 0.01
    l1_weight: float = 0.001
    proximal_mu: float = 0.1
    elastic_alpha: float = 0.5
    
    # Dropout
    use_dropout: bool = True
    dropout_rate: float = 0.3
    
    # Model
    input_dim: int = 32
    hidden_dim: int = 128
    num_classes: int = 10


class RegDataset(Dataset):
    """Dataset for regularization experiments."""
    
    def __init__(self, features: np.ndarray, labels: np.ndarray):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self): return len(self.labels)
    def __getitem__(self, idx): return self.features[idx], self.labels[idx]


class RegularizedModel(nn.Module):
    """Model with regularization support."""
    
    def __init__(self, config: RegConfig):
        super().__init__()
        self.config = config
        
        layers = [
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.ReLU(),
        ]
        
        if config.use_dropout:
            layers.append(nn.Dropout(config.dropout_rate))
        
        layers.extend([
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
        ])
        
        if config.use_dropout:
            layers.append(nn.Dropout(config.dropout_rate))
        
        layers.append(nn.Linear(config.hidden_dim // 2, config.num_classes))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
    
    def get_l2_regularization(self) -> torch.Tensor:
        """Compute L2 regularization term."""
        l2_reg = torch.tensor(0.0)
        for param in self.parameters():
            l2_reg += torch.sum(param ** 2)
        return l2_reg
    
    def get_l1_regularization(self) -> torch.Tensor:
        """Compute L1 regularization term."""
        l1_reg = torch.tensor(0.0)
        for param in self.parameters():
            l1_reg += torch.sum(torch.abs(param))
        return l1_reg
    
    def get_elastic_regularization(self, alpha: float = 0.5) -> torch.Tensor:
        """Compute elastic net regularization."""
        l1 = self.get_l1_regularization()
        l2 = self.get_l2_regularization()
        return alpha * l1 + (1 - alpha) * l2


class ProximalRegularizer:
    """Proximal regularization for FedProx."""
    
    def __init__(self, global_model: nn.Module, mu: float = 0.1):
        self.global_params = {
            name: param.clone().detach()
            for name, param in global_model.named_parameters()
        }
        self.mu = mu
    
    def get_proximal_term(self, local_model: nn.Module) -> torch.Tensor:
        """Compute proximal regularization term."""
        prox_term = torch.tensor(0.0)
        
        for name, param in local_model.named_parameters():
            if name in self.global_params:
                diff = param - self.global_params[name]
                prox_term += torch.sum(diff ** 2)
        
        return (self.mu / 2) * prox_term
    
    def update_global(self, global_model: nn.Module) -> None:
        """Update reference to global model."""
        self.global_params = {
            name: param.clone().detach()
            for name, param in global_model.named_parameters()
        }
```

### Part 2: FL System with Regularization

```python
def create_heterogeneous_data(config: RegConfig) -> Tuple[List[RegDataset], RegDataset]:
    """Create non-IID data for regularization experiments."""
    np.random.seed(config.seed)
    
    datasets = []
    for i in range(config.num_clients):
        # Each client has biased class distribution
        dominant_classes = np.random.choice(config.num_classes, 3, replace=False)
        
        n = np.random.randint(50, 150)
        x = np.random.randn(n, config.input_dim).astype(np.float32)
        
        # 70% from dominant classes
        n_dominant = int(0.7 * n)
        y = np.zeros(n, dtype=np.int64)
        y[:n_dominant] = np.random.choice(dominant_classes, n_dominant)
        y[n_dominant:] = np.random.randint(0, config.num_classes, n - n_dominant)
        np.random.shuffle(y)
        
        # Add signal
        for j in range(n):
            x[j, y[j] % config.input_dim] += 2.0
        
        datasets.append(RegDataset(x, y))
    
    # Balanced test set
    test_x = np.random.randn(500, config.input_dim).astype(np.float32)
    test_y = np.random.randint(0, config.num_classes, 500)
    for j in range(500):
        test_x[j, test_y[j] % config.input_dim] += 2.0
    
    return datasets, RegDataset(test_x, test_y)


class RegularizedClient:
    """FL client with regularization support."""
    
    def __init__(self, client_id: int, dataset: RegDataset, config: RegConfig):
        self.client_id = client_id
        self.dataset = dataset
        self.config = config
    
    def train(self, model: nn.Module) -> Dict[str, Any]:
        """Train with regularization."""
        local = copy.deepcopy(model)
        
        # Create proximal regularizer if needed
        prox_reg = None
        if self.config.reg_type in [RegularizationType.PROXIMAL, RegularizationType.COMBINED]:
            prox_reg = ProximalRegularizer(model, self.config.proximal_mu)
        
        # Optimizer with weight decay if using L2
        weight_decay = self.config.l2_weight if self.config.reg_type == RegularizationType.L2 else 0
        opt = torch.optim.SGD(local.parameters(), lr=self.config.learning_rate, weight_decay=weight_decay)
        
        loader = DataLoader(self.dataset, batch_size=self.config.batch_size, shuffle=True)
        
        local.train()
        total_loss, n_batches = 0, 0
        
        for _ in range(self.config.local_epochs):
            for x, y in loader:
                opt.zero_grad()
                
                # Task loss
                outputs = local(x)
                loss = F.cross_entropy(outputs, y)
                
                # Add regularization terms
                if self.config.reg_type == RegularizationType.L1:
                    loss += self.config.l1_weight * local.get_l1_regularization()
                
                elif self.config.reg_type == RegularizationType.ELASTIC:
                    loss += self.config.l2_weight * local.get_elastic_regularization(
                        self.config.elastic_alpha
                    )
                
                elif self.config.reg_type == RegularizationType.PROXIMAL:
                    loss += prox_reg.get_proximal_term(local)
                
                elif self.config.reg_type == RegularizationType.COMBINED:
                    loss += self.config.l2_weight * local.get_l2_regularization()
                    loss += prox_reg.get_proximal_term(local)
                
                loss.backward()
                opt.step()
                
                total_loss += loss.item()
                n_batches += 1
        
        return {
            "state_dict": {k: v.cpu() for k, v in local.state_dict().items()},
            "num_samples": len(self.dataset),
            "loss": total_loss / n_batches if n_batches > 0 else 0,
        }


class RegularizedServer:
    """FL server for regularization experiments."""
    
    def __init__(self, model: nn.Module, clients: List[RegularizedClient],
                 test_dataset: RegDataset, config: RegConfig):
        self.model = model
        self.clients = clients
        self.test_dataset = test_dataset
        self.config = config
        self.history = []
    
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
            selected = np.random.choice(self.clients, min(self.config.clients_per_round, len(self.clients)), replace=False)
            updates = [c.train(self.model) for c in selected]
            self.aggregate(updates)
            
            acc, loss = self.evaluate()
            self.history.append({"round": r, "accuracy": acc, "loss": loss})
            
            if (r + 1) % 10 == 0:
                print(f"Round {r+1}: acc={acc:.4f}, loss={loss:.4f}")
        
        return self.history


def compare_regularization():
    """Compare different regularization strategies."""
    results = {}
    
    for reg_type in [RegularizationType.NONE, RegularizationType.L2, RegularizationType.PROXIMAL, RegularizationType.COMBINED]:
        config = RegConfig(num_rounds=50, num_clients=10, reg_type=reg_type)
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        
        datasets, test = create_heterogeneous_data(config)
        clients = [RegularizedClient(i, d, config) for i, d in enumerate(datasets)]
        model = RegularizedModel(config)
        server = RegularizedServer(model, clients, test, config)
        
        print(f"\n=== {reg_type.value.upper()} ===")
        history = server.train()
        results[reg_type.value] = history[-1]["accuracy"]
    
    print("\n=== Results ===")
    for name, acc in results.items():
        print(f"{name}: {acc:.4f}")


if __name__ == "__main__":
    compare_regularization()
```

---

## Exercises

1. **Exercise 1**: Implement spectral normalization.
2. **Exercise 2**: Add mixup augmentation.
3. **Exercise 3**: Compare with label smoothing.
4. **Exercise 4**: Implement stochastic depth.
5. **Exercise 5**: Add knowledge distillation regularization.

---

## References

1. Li, T., et al. (2020). Federated optimization in heterogeneous networks (FedProx). *MLSys*.
2. Srivastava, N., et al. (2014). Dropout: A simple way to prevent overfitting. *JMLR*.
3. Krogh, A., & Hertz, J. (1992). A simple weight decay can improve generalization. *NeurIPS*.
4. Acar, D.A.E., et al. (2021). Federated learning based on dynamic regularization. In *ICLR*.
5. Li, Q., et al. (2021). Model-contrastive federated learning. In *CVPR*.

---

*Copyright 2026 Olaf Yunus Laitinen Imanov and Contributors. Released under EUPL 1.2.*
