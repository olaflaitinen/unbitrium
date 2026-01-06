# Tutorial 152: FL Common Mistakes

---

## Metadata

| Property | Value |
|----------|-------|
| **Tutorial ID** | 152 |
| **Title** | Federated Learning Common Mistakes |
| **Category** | Guidelines |
| **Difficulty** | Intermediate |
| **Duration** | 90 minutes |
| **Prerequisites** | Tutorial 001-151 |
| **Author** | Unbitrium Contributors |
| **Last Updated** | January 2026 |

---

## Learning Objectives

By completing this tutorial, you will be able to:

1. **Identify** common mistakes in FL implementations
2. **Understand** why these mistakes occur
3. **Implement** correct solutions
4. **Avoid** pitfalls in production systems
5. **Debug** FL training issues effectively

---

## Prerequisites

Before starting this tutorial, ensure you have:

- Completed tutorials 001-151
- Experience with FL implementation
- Understanding of distributed systems
- Familiarity with debugging techniques

---

## Background and Theory

### Why Mistakes Matter in FL

Federated learning is inherently more complex than centralized ML due to:
- Distributed nature of computation
- Communication constraints
- Data heterogeneity
- Privacy requirements

### Common Mistake Categories

```
FL Mistake Categories:
├── Implementation Errors
│   ├── Incorrect aggregation
│   ├── State management bugs
│   └── Synchronization issues
├── Design Flaws
│   ├── Poor client selection
│   ├── Wrong hyperparameters
│   └── Insufficient robustness
├── Data Issues
│   ├── Ignoring heterogeneity
│   ├── Data leakage
│   └── Label imbalance
└── System Problems
    ├── Memory leaks
    ├── Communication failures
    └── Scalability bottlenecks
```

### Impact of Mistakes

| Mistake Type | Impact | Detection Difficulty |
|--------------|--------|---------------------|
| Aggregation bugs | Model divergence | Medium |
| Data leakage | Privacy violations | Hard |
| Memory issues | System crashes | Easy |
| Hyperparameter errors | Poor convergence | Medium |

---

## Implementation Code

```python
#!/usr/bin/env python3
"""
Tutorial 152: Federated Learning Common Mistakes

This module demonstrates common FL mistakes and their corrections,
providing both incorrect and correct implementations for learning.

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
import copy
import warnings


@dataclass
class MistakeConfig:
    """Configuration for mistake demonstrations."""
    num_rounds: int = 30
    num_clients: int = 10
    input_dim: int = 32
    num_classes: int = 10
    learning_rate: float = 0.01
    batch_size: int = 32
    local_epochs: int = 3
    seed: int = 42


class MistakeDataset(Dataset):
    """Standard dataset for demonstrations."""
    
    def __init__(
        self,
        n: int = 100,
        dim: int = 32,
        classes: int = 10,
        seed: int = 0
    ):
        np.random.seed(seed)
        self.x = torch.randn(n, dim)
        self.y = torch.randint(0, classes, (n,))
        
        # Add class-specific patterns
        for i in range(n):
            self.x[i, self.y[i].item() % dim] += 2.0
    
    def __len__(self) -> int:
        return len(self.y)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx]


class SimpleModel(nn.Module):
    """Simple model for demonstrations."""
    
    def __init__(self, config: MistakeConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, config.num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ============================================================
# MISTAKE 1: Incorrect Aggregation Weights
# ============================================================

class WrongAggregationClient:
    """Client with wrong aggregation approach."""
    
    def __init__(self, client_id: int, dataset: MistakeDataset, config: MistakeConfig):
        self.client_id = client_id
        self.dataset = dataset
        self.config = config
    
    def train(self, model: nn.Module) -> Dict:
        local = copy.deepcopy(model)
        optimizer = torch.optim.SGD(local.parameters(), lr=self.config.learning_rate)
        loader = DataLoader(self.dataset, batch_size=self.config.batch_size, shuffle=True)
        
        local.train()
        for _ in range(self.config.local_epochs):
            for x, y in loader:
                optimizer.zero_grad()
                loss = F.cross_entropy(local(x), y)
                loss.backward()
                optimizer.step()
        
        return {
            "state_dict": {k: v.cpu() for k, v in local.state_dict().items()},
            "num_samples": len(self.dataset)
        }


def wrong_aggregation(updates: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    MISTAKE: Simple averaging without sample weighting.
    
    This ignores the number of samples each client has,
    treating all clients equally regardless of data size.
    """
    new_state = {}
    n = len(updates)
    
    for key in updates[0]["state_dict"]:
        # WRONG: Equal weights for all clients
        new_state[key] = sum(
            u["state_dict"][key].float() for u in updates
        ) / n
    
    return new_state


def correct_aggregation(updates: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    CORRECT: Weighted averaging by sample count.
    
    This properly weights each client's contribution
    based on their dataset size.
    """
    total_samples = sum(u["num_samples"] for u in updates)
    new_state = {}
    
    for key in updates[0]["state_dict"]:
        # CORRECT: Weight by number of samples
        new_state[key] = sum(
            (u["num_samples"] / total_samples) * u["state_dict"][key].float()
            for u in updates
        )
    
    return new_state


# ============================================================
# MISTAKE 2: Not Cloning the Model Before Training
# ============================================================

class NoCloneClient:
    """Client that modifies global model directly (WRONG)."""
    
    def __init__(self, client_id: int, dataset: MistakeDataset, config: MistakeConfig):
        self.client_id = client_id
        self.dataset = dataset
        self.config = config
    
    def train_wrong(self, model: nn.Module) -> Dict:
        """
        MISTAKE: Training directly on the global model.
        
        This modifies the shared model, causing interference
        between clients during parallel training.
        """
        # WRONG: No copy, modifying global model directly
        optimizer = torch.optim.SGD(model.parameters(), lr=self.config.learning_rate)
        loader = DataLoader(self.dataset, batch_size=self.config.batch_size, shuffle=True)
        
        model.train()
        for _ in range(self.config.local_epochs):
            for x, y in loader:
                optimizer.zero_grad()
                loss = F.cross_entropy(model(x), y)
                loss.backward()
                optimizer.step()
        
        return {
            "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
            "num_samples": len(self.dataset)
        }
    
    def train_correct(self, model: nn.Module) -> Dict:
        """
        CORRECT: Deep copy before training.
        
        Creates an independent copy to train locally
        without affecting other clients.
        """
        # CORRECT: Deep copy the model
        local = copy.deepcopy(model)
        optimizer = torch.optim.SGD(local.parameters(), lr=self.config.learning_rate)
        loader = DataLoader(self.dataset, batch_size=self.config.batch_size, shuffle=True)
        
        local.train()
        for _ in range(self.config.local_epochs):
            for x, y in loader:
                optimizer.zero_grad()
                loss = F.cross_entropy(local(x), y)
                loss.backward()
                optimizer.step()
        
        return {
            "state_dict": {k: v.cpu() for k, v in local.state_dict().items()},
            "num_samples": len(self.dataset)
        }


# ============================================================
# MISTAKE 3: Ignoring Data Heterogeneity
# ============================================================

def create_iid_data(
    num_clients: int,
    samples_per_client: int,
    config: MistakeConfig
) -> List[MistakeDataset]:
    """Create IID data across clients."""
    return [
        MistakeDataset(n=samples_per_client, seed=i)
        for i in range(num_clients)
    ]


def create_non_iid_data(
    num_clients: int,
    samples_per_client: int,
    config: MistakeConfig
) -> List[MistakeDataset]:
    """
    Create non-IID data with label skew.
    
    Each client has data biased towards specific classes.
    """
    datasets = []
    classes_per_client = config.num_classes // num_clients
    
    for i in range(num_clients):
        np.random.seed(config.seed + i)
        x = torch.randn(samples_per_client, config.input_dim)
        
        # Assign labels biased to specific classes
        main_classes = [
            (i * classes_per_client + j) % config.num_classes
            for j in range(classes_per_client)
        ]
        
        y = torch.tensor([
            np.random.choice(main_classes) if np.random.random() < 0.8
            else np.random.randint(0, config.num_classes)
            for _ in range(samples_per_client)
        ])
        
        # Create dataset
        ds = MistakeDataset(n=samples_per_client, seed=i)
        ds.y = y
        datasets.append(ds)
    
    return datasets


# ============================================================
# MISTAKE 4: No Gradient Clipping
# ============================================================

class NoClippingClient:
    """Client without gradient clipping (can cause issues)."""
    
    def __init__(self, client_id: int, dataset: MistakeDataset, config: MistakeConfig):
        self.client_id = client_id
        self.dataset = dataset
        self.config = config
    
    def train_no_clipping(self, model: nn.Module) -> Dict:
        """
        MISTAKE: No gradient clipping.
        
        Can lead to exploding gradients, especially
        with heterogeneous data.
        """
        local = copy.deepcopy(model)
        optimizer = torch.optim.SGD(local.parameters(), lr=self.config.learning_rate)
        loader = DataLoader(self.dataset, batch_size=self.config.batch_size, shuffle=True)
        
        local.train()
        for _ in range(self.config.local_epochs):
            for x, y in loader:
                optimizer.zero_grad()
                loss = F.cross_entropy(local(x), y)
                loss.backward()
                # WRONG: No gradient clipping
                optimizer.step()
        
        return {
            "state_dict": {k: v.cpu() for k, v in local.state_dict().items()},
            "num_samples": len(self.dataset)
        }
    
    def train_with_clipping(self, model: nn.Module, max_norm: float = 1.0) -> Dict:
        """
        CORRECT: Apply gradient clipping.
        
        Prevents exploding gradients and improves stability.
        """
        local = copy.deepcopy(model)
        optimizer = torch.optim.SGD(local.parameters(), lr=self.config.learning_rate)
        loader = DataLoader(self.dataset, batch_size=self.config.batch_size, shuffle=True)
        
        local.train()
        for _ in range(self.config.local_epochs):
            for x, y in loader:
                optimizer.zero_grad()
                loss = F.cross_entropy(local(x), y)
                loss.backward()
                # CORRECT: Clip gradients
                torch.nn.utils.clip_grad_norm_(local.parameters(), max_norm)
                optimizer.step()
        
        return {
            "state_dict": {k: v.cpu() for k, v in local.state_dict().items()},
            "num_samples": len(self.dataset)
        }


# ============================================================
# MISTAKE 5: Memory Leaks from Not Detaching Tensors
# ============================================================

class MemoryLeakClient:
    """Client with potential memory leak."""
    
    def __init__(self, client_id: int, dataset: MistakeDataset, config: MistakeConfig):
        self.client_id = client_id
        self.dataset = dataset
        self.config = config
        self.history: List[torch.Tensor] = []  # Potential leak source
    
    def train_with_leak(self, model: nn.Module) -> Dict:
        """
        MISTAKE: Storing tensors without detaching.
        
        Keeps computation graph in memory, causing leaks.
        """
        local = copy.deepcopy(model)
        optimizer = torch.optim.SGD(local.parameters(), lr=self.config.learning_rate)
        loader = DataLoader(self.dataset, batch_size=self.config.batch_size, shuffle=True)
        
        local.train()
        for _ in range(self.config.local_epochs):
            for x, y in loader:
                optimizer.zero_grad()
                output = local(x)
                loss = F.cross_entropy(output, y)
                loss.backward()
                optimizer.step()
                
                # WRONG: Storing tensor with grad history
                self.history.append(loss)
        
        return {
            "state_dict": {k: v.cpu() for k, v in local.state_dict().items()},
            "num_samples": len(self.dataset)
        }
    
    def train_no_leak(self, model: nn.Module) -> Dict:
        """
        CORRECT: Detach tensors before storing.
        
        Frees computation graph from memory.
        """
        local = copy.deepcopy(model)
        optimizer = torch.optim.SGD(local.parameters(), lr=self.config.learning_rate)
        loader = DataLoader(self.dataset, batch_size=self.config.batch_size, shuffle=True)
        
        local.train()
        losses = []
        for _ in range(self.config.local_epochs):
            for x, y in loader:
                optimizer.zero_grad()
                output = local(x)
                loss = F.cross_entropy(output, y)
                loss.backward()
                optimizer.step()
                
                # CORRECT: Detach and convert to Python float
                losses.append(loss.item())
        
        return {
            "state_dict": {k: v.cpu() for k, v in local.state_dict().items()},
            "num_samples": len(self.dataset),
            "avg_loss": np.mean(losses)
        }


# ============================================================
# MISTAKE 6: Not Setting Model to Eval Mode
# ============================================================

def evaluate_wrong(model: nn.Module, test_data: MistakeDataset) -> float:
    """
    MISTAKE: Not setting model to eval mode.
    
    Dropout and BatchNorm behave differently in training mode,
    leading to inconsistent evaluation results.
    """
    # WRONG: Model stays in training mode
    loader = DataLoader(test_data, batch_size=64)
    correct = 0
    total = 0
    
    with torch.no_grad():
        for x, y in loader:
            output = model(x)
            pred = output.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += len(y)
    
    return correct / total


def evaluate_correct(model: nn.Module, test_data: MistakeDataset) -> float:
    """
    CORRECT: Set model to eval mode before evaluation.
    
    Ensures consistent behavior of Dropout and BatchNorm.
    """
    # CORRECT: Set to eval mode
    model.eval()
    loader = DataLoader(test_data, batch_size=64)
    correct = 0
    total = 0
    
    with torch.no_grad():
        for x, y in loader:
            output = model(x)
            pred = output.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += len(y)
    
    return correct / total


# ============================================================
# Complete Correct Implementation
# ============================================================

class CorrectClient:
    """Client with all correct implementations."""
    
    def __init__(
        self,
        client_id: int,
        dataset: MistakeDataset,
        config: MistakeConfig
    ):
        self.client_id = client_id
        self.dataset = dataset
        self.config = config
    
    def train(self, model: nn.Module) -> Dict:
        """Correct training implementation."""
        # 1. Deep copy model
        local = copy.deepcopy(model)
        
        # 2. Use appropriate optimizer
        optimizer = torch.optim.SGD(
            local.parameters(),
            lr=self.config.learning_rate,
            momentum=0.9
        )
        
        loader = DataLoader(
            self.dataset,
            batch_size=self.config.batch_size,
            shuffle=True
        )
        
        local.train()
        losses = []
        
        for _ in range(self.config.local_epochs):
            for x, y in loader:
                optimizer.zero_grad()
                output = local(x)
                loss = F.cross_entropy(output, y)
                loss.backward()
                
                # 3. Clip gradients
                torch.nn.utils.clip_grad_norm_(local.parameters(), 1.0)
                
                optimizer.step()
                
                # 4. Properly store loss
                losses.append(loss.item())
        
        return {
            "state_dict": {k: v.cpu() for k, v in local.state_dict().items()},
            "num_samples": len(self.dataset),
            "avg_loss": np.mean(losses)
        }


class CorrectServer:
    """Server with all correct implementations."""
    
    def __init__(
        self,
        model: nn.Module,
        clients: List[CorrectClient],
        test_data: MistakeDataset,
        config: MistakeConfig
    ):
        self.model = model
        self.clients = clients
        self.test_data = test_data
        self.config = config
    
    def aggregate(self, updates: List[Dict]) -> None:
        """Correct weighted aggregation."""
        if not updates:
            return
        
        total_samples = sum(u["num_samples"] for u in updates)
        new_state = {}
        
        for key in updates[0]["state_dict"]:
            new_state[key] = sum(
                (u["num_samples"] / total_samples) * u["state_dict"][key].float()
                for u in updates
            )
        
        self.model.load_state_dict(new_state)
    
    def evaluate(self) -> float:
        """Correct evaluation."""
        self.model.eval()
        loader = DataLoader(self.test_data, batch_size=64)
        correct = 0
        total = 0
        
        with torch.no_grad():
            for x, y in loader:
                output = self.model(x)
                pred = output.argmax(dim=1)
                correct += (pred == y).sum().item()
                total += len(y)
        
        return correct / total
    
    def train(self) -> List[Dict]:
        """Run correct federated training."""
        history = []
        
        for round_num in range(self.config.num_rounds):
            # Collect updates
            updates = [c.train(self.model) for c in self.clients]
            
            # Aggregate with proper weighting
            self.aggregate(updates)
            
            # Evaluate with model in eval mode
            accuracy = self.evaluate()
            avg_loss = np.mean([u["avg_loss"] for u in updates])
            
            history.append({
                "round": round_num,
                "accuracy": accuracy,
                "avg_loss": avg_loss
            })
            
            if (round_num + 1) % 10 == 0:
                print(f"Round {round_num + 1}: acc={accuracy:.4f}, loss={avg_loss:.4f}")
        
        return history


def main():
    """Main demonstration."""
    print("=" * 60)
    print("Tutorial 152: FL Common Mistakes")
    print("=" * 60)
    
    config = MistakeConfig()
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    
    # Create data and clients
    datasets = [MistakeDataset(seed=i) for i in range(config.num_clients)]
    clients = [CorrectClient(i, d, config) for i, d in enumerate(datasets)]
    test_data = MistakeDataset(n=200, seed=999)
    
    # Create model
    model = SimpleModel(config)
    
    # Train with correct implementation
    server = CorrectServer(model, clients, test_data, config)
    
    print("\nRunning correct FL implementation...")
    print("-" * 40)
    history = server.train()
    
    print("\n" + "=" * 60)
    print("Training Complete")
    print(f"Final Accuracy: {history[-1]['accuracy']:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
```

---

## Mistake Summary Checklist

### Before Every FL Project

- [ ] Always deep copy models before local training
- [ ] Use weighted aggregation based on sample counts
- [ ] Apply gradient clipping for stability
- [ ] Detach tensors before storing
- [ ] Set model to eval mode for evaluation
- [ ] Handle data heterogeneity appropriately
- [ ] Test with both IID and non-IID data

---

## Exercises

1. **Exercise 1**: Create a memory profiler to detect leaks
2. **Exercise 2**: Implement a validator for FL configurations
3. **Exercise 3**: Build a debugging tool for aggregation issues
4. **Exercise 4**: Create unit tests for each mistake scenario

---

## References

1. Li, T., et al. (2020). Federated optimization in heterogeneous networks. In *MLSys*.
2. Kairouz, P., et al. (2021). Advances and open problems in federated learning. *FnTML*.
3. McMahan, B., et al. (2017). Communication-efficient learning of deep networks. In *AISTATS*.

---

*Copyright 2026 Olaf Yunus Laitinen Imanov and Contributors. Released under EUPL 1.2.*
