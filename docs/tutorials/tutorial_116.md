# Tutorial 116: FL for Finance

---

## Metadata

| Property | Value |
|----------|-------|
| **Tutorial ID** | 116 |
| **Title** | Federated Learning for Finance Applications |
| **Category** | Domain Applications |
| **Difficulty** | Advanced |
| **Duration** | 120 minutes |
| **Prerequisites** | Tutorial 001-115 |
| **Author** | Unbitrium Contributors |
| **Last Updated** | January 2026 |

---

## Learning Objectives

By completing this tutorial, you will be able to:

1. **Understand** FL applications in finance
2. **Implement** federated credit scoring
3. **Design** privacy-preserving risk models
4. **Analyze** regulatory compliance
5. **Deploy** FL for financial institutions

---

## Prerequisites

Before starting this tutorial, ensure you have:

- Completed tutorials 001-115
- Understanding of FL fundamentals
- Knowledge of finance ML applications
- Familiarity with imbalanced learning

---

## Background and Theory

### Finance FL Use Cases

Financial applications of FL include:
- Cross-bank fraud detection
- Credit risk assessment
- Anti-money laundering (AML)
- Algorithmic trading signals
- Customer churn prediction

### Architectural Patterns

```
Finance FL Architecture:
├── Bank/Institution Level
│   ├── Transaction data
│   ├── Customer profiles
│   ├── Risk assessments
│   └── Local training
├── Consortium Level
│   ├── Secure aggregation
│   ├── Compliance layer
│   └── Audit logging
└── Regulatory Interface
    ├── Model cards
    ├── Explainability
    └── Fairness metrics
```

### Challenges

| Challenge | Description | Mitigation |
|-----------|-------------|------------|
| Data Imbalance | <1% fraud rate | Weighted loss |
| Privacy | Strict regulations | DP, SecAgg |
| Explainability | Regulatory need | Attention, SHAP |
| Fairness | Bias concerns | Fair FL |

---

## Implementation Code

```python
#!/usr/bin/env python3
"""
Tutorial 116: Federated Learning for Finance

This module implements FL for financial applications with
credit scoring and risk assessment.

Copyright 2026 Olaf Yunus Laitinen Imanov and Contributors
Released under EUPL 1.2
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import copy
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class FinanceAppConfig:
    """Configuration for financial FL applications."""
    
    num_rounds: int = 50
    num_institutions: int = 8
    institutions_per_round: int = 6
    
    num_features: int = 30
    hidden_dim: int = 64
    num_classes: int = 2  # Good/Bad credit
    
    learning_rate: float = 0.001
    batch_size: int = 64
    local_epochs: int = 5
    
    default_rate: float = 0.15  # 15% default rate
    
    seed: int = 42


class CreditDataset(Dataset):
    """Credit scoring dataset."""
    
    def __init__(
        self,
        institution_id: int,
        n: int = 1000,
        num_features: int = 30,
        default_rate: float = 0.15,
        seed: int = 0
    ):
        np.random.seed(seed + institution_id)
        
        self.institution_id = institution_id
        
        # Generate credit features
        self.x, self.y = self._generate_data(n, num_features, default_rate)
    
    def _generate_data(
        self,
        n: int,
        features: int,
        default_rate: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate synthetic credit data."""
        x_list = []
        y_list = []
        
        for _ in range(n):
            is_default = np.random.random() < default_rate
            x = self._generate_customer(is_default, features)
            x_list.append(x)
            y_list.append(1 if is_default else 0)
        
        return (
            torch.tensor(np.array(x_list), dtype=torch.float32),
            torch.tensor(y_list, dtype=torch.long)
        )
    
    def _generate_customer(
        self,
        is_default: bool,
        features: int
    ) -> np.ndarray:
        """Generate customer features based on risk."""
        x = np.zeros(features)
        
        if is_default:
            # High-risk profile
            x[0] = np.random.uniform(0.4, 0.9)   # High utilization
            x[1] = np.random.uniform(0, 0.4)    # Low income ratio
            x[2] = np.random.poisson(3)          # Multiple delinquencies
            x[3] = np.random.uniform(0, 3)       # Short credit history
            x[4] = np.random.uniform(0.1, 0.3)   # Low savings
        else:
            # Low-risk profile
            x[0] = np.random.uniform(0.1, 0.4)   # Low utilization
            x[1] = np.random.uniform(0.3, 0.8)   # Good income ratio
            x[2] = np.random.poisson(0.2)        # Few delinquencies
            x[3] = np.random.uniform(5, 30)      # Long credit history
            x[4] = np.random.uniform(0.3, 0.8)   # Good savings
        
        # Additional features
        for i in range(5, features):
            x[i] = np.random.randn() + (0.5 if is_default else -0.5)
        
        return (x - x.mean()) / (x.std() + 1e-8)
    
    def __len__(self) -> int:
        return len(self.y)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx]


class CreditScoringModel(nn.Module):
    """Credit scoring model with interpretability."""
    
    def __init__(self, config: FinanceAppConfig):
        super().__init__()
        
        self.attention = nn.Sequential(
            nn.Linear(config.num_features, config.num_features),
            nn.Tanh(),
            nn.Linear(config.num_features, config.num_features),
            nn.Softmax(dim=-1)
        )
        
        self.encoder = nn.Sequential(
            nn.Linear(config.num_features, config.hidden_dim),
            nn.BatchNorm1d(config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
        )
        
        self.classifier = nn.Linear(config.hidden_dim // 2, config.num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn = self.attention(x)
        attended = x * attn
        features = self.encoder(attended)
        return self.classifier(features)
    
    def get_risk_score(self, x: torch.Tensor) -> torch.Tensor:
        """Get default probability."""
        logits = self.forward(x)
        probs = F.softmax(logits, dim=-1)
        return probs[:, 1]  # Probability of default


class InstitutionClient:
    """FL client for a financial institution."""
    
    def __init__(
        self,
        institution_id: int,
        dataset: CreditDataset,
        config: FinanceAppConfig
    ):
        self.institution_id = institution_id
        self.dataset = dataset
        self.config = config
        
        # Compute class weights
        num_default = (dataset.y == 1).sum().item()
        num_good = len(dataset.y) - num_default
        self.class_weights = torch.tensor([
            1.0,
            num_good / max(num_default, 1)
        ])
    
    def train(self, model: nn.Module) -> Dict[str, Any]:
        """Train on local credit data."""
        local_model = copy.deepcopy(model)
        optimizer = torch.optim.Adam(
            local_model.parameters(),
            lr=self.config.learning_rate
        )
        
        # Weighted sampling
        sample_weights = torch.zeros(len(self.dataset))
        for i in range(len(self.dataset)):
            sample_weights[i] = self.class_weights[self.dataset.y[i]]
        
        sampler = WeightedRandomSampler(
            sample_weights, len(self.dataset), replacement=True
        )
        
        loader = DataLoader(
            self.dataset,
            batch_size=self.config.batch_size,
            sampler=sampler
        )
        
        local_model.train()
        total_loss = 0.0
        num_batches = 0
        
        for _ in range(self.config.local_epochs):
            for x, y in loader:
                optimizer.zero_grad()
                output = local_model(x)
                loss = F.cross_entropy(output, y, weight=self.class_weights)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(local_model.parameters(), 1.0)
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
        
        return {
            "state_dict": {k: v.cpu() for k, v in local_model.state_dict().items()},
            "num_samples": len(self.dataset),
            "avg_loss": total_loss / num_batches,
            "institution_id": self.institution_id
        }


class FinanceConsortiumServer:
    """Server for financial consortium."""
    
    def __init__(
        self,
        model: nn.Module,
        institutions: List[InstitutionClient],
        test_data: CreditDataset,
        config: FinanceAppConfig
    ):
        self.model = model
        self.institutions = institutions
        self.test_data = test_data
        self.config = config
        self.history: List[Dict] = []
    
    def aggregate(self, updates: List[Dict]) -> None:
        """FedAvg aggregation."""
        total_samples = sum(u["num_samples"] for u in updates)
        new_state = {}
        
        for key in updates[0]["state_dict"]:
            new_state[key] = sum(
                (u["num_samples"] / total_samples) * u["state_dict"][key].float()
                for u in updates
            )
        
        self.model.load_state_dict(new_state)
    
    def evaluate(self) -> Dict[str, float]:
        """Evaluate credit model."""
        self.model.eval()
        loader = DataLoader(self.test_data, batch_size=64)
        
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for x, y in loader:
                output = self.model(x)
                all_preds.append(output)
                all_targets.append(y)
        
        preds = torch.cat(all_preds)
        targets = torch.cat(all_targets)
        
        pred_labels = preds.argmax(dim=1)
        accuracy = (pred_labels == targets).float().mean().item()
        
        # Default detection metrics
        tp = ((pred_labels == 1) & (targets == 1)).sum().item()
        fp = ((pred_labels == 1) & (targets == 0)).sum().item()
        fn = ((pred_labels == 0) & (targets == 1)).sum().item()
        
        precision = tp / (tp + fp) if tp + fp > 0 else 0
        recall = tp / (tp + fn) if tp + fn > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
        
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }
    
    def train(self) -> List[Dict]:
        """Run FL training."""
        logger.info(f"Starting finance FL with {len(self.institutions)} institutions")
        
        for round_num in range(self.config.num_rounds):
            n = min(self.config.institutions_per_round, len(self.institutions))
            indices = np.random.choice(len(self.institutions), n, replace=False)
            selected = [self.institutions[i] for i in indices]
            
            updates = [inst.train(self.model) for inst in selected]
            self.aggregate(updates)
            
            metrics = self.evaluate()
            
            record = {"round": round_num, **metrics}
            self.history.append(record)
            
            if (round_num + 1) % 10 == 0:
                logger.info(
                    f"Round {round_num + 1}: "
                    f"acc={metrics['accuracy']:.4f}, "
                    f"f1={metrics['f1']:.4f}"
                )
        
        return self.history


def main():
    """Main entry point."""
    print("=" * 60)
    print("Tutorial 116: FL for Finance")
    print("=" * 60)
    
    config = FinanceAppConfig()
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    
    # Create institutions
    institutions = []
    for i in range(config.num_institutions):
        dataset = CreditDataset(
            institution_id=i,
            n=1000,
            num_features=config.num_features,
            default_rate=config.default_rate * np.random.uniform(0.7, 1.3),
            seed=config.seed
        )
        client = InstitutionClient(i, dataset, config)
        institutions.append(client)
    
    test_data = CreditDataset(institution_id=999, n=500, seed=999)
    model = CreditScoringModel(config)
    
    server = FinanceConsortiumServer(model, institutions, test_data, config)
    history = server.train()
    
    print("\n" + "=" * 60)
    print("Training Complete")
    print(f"Final F1: {history[-1]['f1']:.4f}")
    print(f"Final Recall: {history[-1]['recall']:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
```

---

## Key Insights

### Finance FL Best Practices

1. **Handle Imbalance**: Use weighted sampling and loss
2. **Interpretability**: Add attention for feature importance
3. **Compliance**: Implement audit logging
4. **Fairness**: Monitor for demographic bias

---

## Exercises

1. **Exercise 1**: Add differential privacy
2. **Exercise 2**: Implement AML detection
3. **Exercise 3**: Add fairness constraints
4. **Exercise 4**: Create model cards

---

## References

1. Long, G., et al. (2020). Federated learning for open banking. In *IJCAI*.
2. Yang, W., et al. (2019). FFD: Federated fraud detection. In *BigData*.

---

*Copyright 2026 Olaf Yunus Laitinen Imanov and Contributors. Released under EUPL 1.2.*
