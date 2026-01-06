# Tutorial 125: FL Interpretability

---

## Metadata

| Property | Value |
|----------|-------|
| **Tutorial ID** | 125 |
| **Title** | Federated Learning Interpretability |
| **Category** | Explainability |
| **Difficulty** | Advanced |
| **Duration** | 120 minutes |
| **Prerequisites** | Tutorial 001-124 |
| **Author** | Unbitrium Contributors |
| **Last Updated** | January 2026 |

---

## Learning Objectives

By completing this tutorial, you will be able to:

1. **Understand** interpretability challenges in FL
2. **Implement** federated feature importance
3. **Design** explainable FL models
4. **Analyze** client-level explanations
5. **Deploy** interpretable FL systems

---

## Prerequisites

Before starting this tutorial, ensure you have:

- Completed tutorials 001-124
- Understanding of FL fundamentals
- Knowledge of ML interpretability
- Familiarity with gradient-based methods

---

## Background and Theory

### Interpretability in FL

FL creates unique interpretability challenges:
- Global model, local data patterns
- Privacy limits explanation methods
- Heterogeneous client behaviors
- Distributed feature importance

### Interpretability Methods

```
FL Interpretability Methods:
├── Feature-Level
│   ├── Gradient-based importance
│   ├── Integrated gradients
│   └── Attention weights
├── Model-Level
│   ├── Global explanations
│   ├── Layer analysis
│   └── Concept activation
├── Client-Level
│   ├── Per-client importance
│   ├── Contribution analysis
│   └── Counterfactuals
└── Privacy-Preserving
    ├── Aggregated explanations
    ├── Differential privacy
    └── Secure computation
```

### Method Comparison

| Method | Privacy | Fidelity | Complexity |
|--------|---------|----------|------------|
| Gradient | High | High | Low |
| SHAP | Medium | Very High | High |
| Attention | High | Medium | Low |
| Counterfactual | Low | High | High |

---

## Implementation Code

```python
#!/usr/bin/env python3
"""
Tutorial 125: Federated Learning Interpretability

This module implements interpretability techniques for
federated learning including feature importance and explanations.

Copyright 2026 Olaf Yunus Laitinen Imanov and Contributors
Released under EUPL 1.2
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import copy
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class InterpretConfig:
    """Configuration for interpretable FL."""
    
    num_rounds: int = 30
    num_clients: int = 10
    clients_per_round: int = 5
    
    input_dim: int = 32
    hidden_dim: int = 64
    num_classes: int = 10
    
    learning_rate: float = 0.01
    batch_size: int = 32
    local_epochs: int = 3
    
    seed: int = 42


class InterpretDataset(Dataset):
    """Dataset with known feature patterns for interpretability testing."""
    
    def __init__(
        self,
        client_id: int,
        n: int = 200,
        dim: int = 32,
        classes: int = 10,
        seed: int = 0
    ):
        np.random.seed(seed + client_id)
        
        self.important_features = [0, 1, 2]  # Known important features
        
        self.x = torch.randn(n, dim, dtype=torch.float32)
        self.y = torch.randint(0, classes, (n,), dtype=torch.long)
        
        # Add strong signal on important features
        for i in range(n):
            for feat in self.important_features:
                self.x[i, feat] = self.y[i].item() * 0.5 + np.random.randn() * 0.1
    
    def __len__(self) -> int:
        return len(self.y)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx]


class InterpretableModel(nn.Module):
    """Model with built-in interpretability."""
    
    def __init__(self, config: InterpretConfig):
        super().__init__()
        
        self.config = config
        
        # Feature attention for interpretability
        self.feature_attention = nn.Sequential(
            nn.Linear(config.input_dim, config.input_dim),
            nn.Tanh(),
            nn.Linear(config.input_dim, config.input_dim),
            nn.Softmax(dim=-1)
        )
        
        # Main network
        self.encoder = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
        )
        
        self.classifier = nn.Linear(config.hidden_dim // 2, config.num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute attention weights
        attn = self.feature_attention(x)
        x_attended = x * attn
        
        # Encode and classify
        features = self.encoder(x_attended)
        return self.classifier(features)
    
    def get_attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        """Get feature attention weights."""
        with torch.no_grad():
            return self.feature_attention(x)


class GradientExplainer:
    """Gradient-based feature importance."""
    
    @staticmethod
    def compute_gradients(
        model: nn.Module,
        x: torch.Tensor,
        target_class: Optional[int] = None
    ) -> torch.Tensor:
        """Compute input gradients."""
        x_grad = x.clone().requires_grad_(True)
        
        output = model(x_grad)
        
        if target_class is None:
            target_class = output.argmax(dim=-1)
        
        # Get score for target class
        if output.dim() == 1:
            score = output[target_class]
        else:
            score = output[range(len(output)), target_class].sum()
        
        score.backward()
        
        return x_grad.grad.abs()
    
    @staticmethod
    def integrated_gradients(
        model: nn.Module,
        x: torch.Tensor,
        baseline: Optional[torch.Tensor] = None,
        steps: int = 50
    ) -> torch.Tensor:
        """Compute integrated gradients."""
        if baseline is None:
            baseline = torch.zeros_like(x)
        
        # Generate interpolation points
        alphas = torch.linspace(0, 1, steps)
        
        grads = []
        for alpha in alphas:
            interp = baseline + alpha * (x - baseline)
            grad = GradientExplainer.compute_gradients(model, interp)
            grads.append(grad)
        
        # Integrate
        avg_grad = torch.stack(grads).mean(dim=0)
        ig = (x - baseline) * avg_grad
        
        return ig.abs()


class FederatedExplainer:
    """Federated explanation aggregator."""
    
    def __init__(self, config: InterpretConfig):
        self.config = config
        self.client_importances: Dict[int, torch.Tensor] = {}
    
    def collect_importance(
        self,
        client_id: int,
        importance: torch.Tensor
    ) -> None:
        """Collect feature importance from client."""
        self.client_importances[client_id] = importance
    
    def aggregate_importance(self) -> torch.Tensor:
        """Aggregate importances across clients."""
        if not self.client_importances:
            return torch.zeros(self.config.input_dim)
        
        # Simple average aggregation
        total = torch.zeros(self.config.input_dim)
        for imp in self.client_importances.values():
            total += imp
        
        return total / len(self.client_importances)
    
    def get_top_features(self, k: int = 5) -> List[Tuple[int, float]]:
        """Get top-k important features."""
        agg = self.aggregate_importance()
        values, indices = torch.topk(agg, k)
        return list(zip(indices.tolist(), values.tolist()))


class InterpretClient:
    """Client with interpretability capabilities."""
    
    def __init__(
        self,
        client_id: int,
        dataset: InterpretDataset,
        config: InterpretConfig
    ):
        self.client_id = client_id
        self.dataset = dataset
        self.config = config
    
    def train(self, model: nn.Module) -> Dict[str, Any]:
        local = copy.deepcopy(model)
        optimizer = torch.optim.SGD(
            local.parameters(),
            lr=self.config.learning_rate
        )
        
        loader = DataLoader(
            self.dataset,
            batch_size=self.config.batch_size,
            shuffle=True
        )
        
        local.train()
        total_loss = 0.0
        num_batches = 0
        
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
            "avg_loss": total_loss / num_batches,
            "client_id": self.client_id
        }
    
    def compute_local_importance(
        self,
        model: nn.Module,
        method: str = "gradient"
    ) -> torch.Tensor:
        """Compute feature importance on local data."""
        model.eval()
        
        # Sample data for explanation
        x, y = self.dataset.x, self.dataset.y
        
        if method == "gradient":
            importance = GradientExplainer.compute_gradients(model, x)
            return importance.mean(dim=0)
        
        elif method == "integrated":
            importance = GradientExplainer.integrated_gradients(model, x)
            return importance.mean(dim=0)
        
        elif method == "attention":
            weights = model.get_attention_weights(x)
            return weights.mean(dim=0)
        
        else:
            raise ValueError(f"Unknown method: {method}")


class InterpretServer:
    """Server with interpretability support."""
    
    def __init__(
        self,
        model: nn.Module,
        clients: List[InterpretClient],
        test_data: InterpretDataset,
        config: InterpretConfig
    ):
        self.model = model
        self.clients = clients
        self.test_data = test_data
        self.config = config
        
        self.explainer = FederatedExplainer(config)
        self.history: List[Dict] = []
    
    def aggregate(self, updates: List[Dict]) -> None:
        """Standard FedAvg aggregation."""
        total_samples = sum(u["num_samples"] for u in updates)
        new_state = {}
        
        for key in updates[0]["state_dict"]:
            new_state[key] = sum(
                (u["num_samples"] / total_samples) * u["state_dict"][key].float()
                for u in updates
            )
        
        self.model.load_state_dict(new_state)
    
    def collect_explanations(self, method: str = "gradient") -> None:
        """Collect explanations from all clients."""
        for client in self.clients:
            importance = client.compute_local_importance(self.model, method)
            self.explainer.collect_importance(client.client_id, importance)
    
    def evaluate(self) -> Dict[str, float]:
        """Evaluate model."""
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
        """Run training with interpretability."""
        logger.info(f"Starting interpretable FL with {len(self.clients)} clients")
        
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
        
        # Final explanations
        self.collect_explanations()
        
        return self.history


def main():
    """Main entry point."""
    print("=" * 60)
    print("Tutorial 125: FL Interpretability")
    print("=" * 60)
    
    config = InterpretConfig()
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    
    # Create clients
    clients = []
    for i in range(config.num_clients):
        dataset = InterpretDataset(client_id=i, dim=config.input_dim, seed=config.seed)
        client = InterpretClient(i, dataset, config)
        clients.append(client)
    
    test_data = InterpretDataset(client_id=999, n=300, seed=999)
    model = InterpretableModel(config)
    
    # Train
    server = InterpretServer(model, clients, test_data, config)
    history = server.train()
    
    # Get explanations
    print("\n" + "=" * 60)
    print("Feature Importance Analysis")
    print("=" * 60)
    
    # Known important features: [0, 1, 2]
    top_features = server.explainer.get_top_features(k=10)
    
    print("\nTop 10 Important Features:")
    for feat_idx, importance in top_features:
        marker = " *" if feat_idx in [0, 1, 2] else ""
        print(f"  Feature {feat_idx}: {importance:.4f}{marker}")
    
    print("\n* = Known important feature")
    print(f"\nFinal Accuracy: {history[-1]['accuracy']:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
```

---

## Key Insights

### Interpretability Best Practices

1. **Use attention**: Built-in interpretability
2. **Aggregate explanations**: Privacy-preserving
3. **Validate with known patterns**: Test interpretability
4. **Multiple methods**: Cross-validate explanations

---

## Exercises

1. **Exercise 1**: Implement LIME for FL
2. **Exercise 2**: Add counterfactual explanations
3. **Exercise 3**: Design client-specific explanations
4. **Exercise 4**: Visualize aggregated importance

---

## References

1. Ribeiro, M.T., et al. (2016). "Why should I trust you?" In *KDD*.
2. Sundararajan, M., et al. (2017). Axiomatic attribution. In *ICML*.
3. Lundberg, S.M., & Lee, S.I. (2017). A unified approach to interpreting. In *NeurIPS*.

---

*Copyright 2026 Olaf Yunus Laitinen Imanov and Contributors. Released under EUPL 1.2.*
