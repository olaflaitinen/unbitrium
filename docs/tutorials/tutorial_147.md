# Tutorial 147: FL Case Study Finance

---

## Metadata

| Property | Value |
|----------|-------|
| **Tutorial ID** | 147 |
| **Title** | Federated Learning Case Study: Finance |
| **Category** | Case Studies |
| **Difficulty** | Advanced |
| **Duration** | 120 minutes |
| **Prerequisites** | Tutorial 001-146 |
| **Author** | Unbitrium Contributors |
| **Last Updated** | January 2026 |

---

## Learning Objectives

By completing this tutorial, you will be able to:

1. **Understand** financial FL applications and regulations
2. **Implement** cross-bank fraud detection FL
3. **Design** privacy-preserving financial models
4. **Analyze** regulatory compliance (GDPR, PCI-DSS)
5. **Deploy** FL for credit risk assessment

---

## Prerequisites

Before starting this tutorial, ensure you have:

- Completed tutorials 001-146
- Understanding of FL fundamentals
- Knowledge of financial data challenges
- Familiarity with imbalanced classification

---

## Background and Theory

### Financial Data Challenges

Financial institutions face unique data challenges:
- Strict regulatory requirements (GDPR, PCI-DSS)
- Competitive sensitivity of data
- Need for cross-institution collaboration
- Highly imbalanced fraud datasets (< 1% fraud)

FL enables financial institutions to:
- Share fraud patterns without data exposure
- Comply with data localization laws
- Improve rare fraud detection
- Build collaborative AML systems

### Financial FL Architecture

```
Financial FL Architecture:
├── Institution Level
│   ├── Transaction databases
│   ├── Customer profiles
│   ├── Risk scores
│   └── Local FL client
├── Consortium Layer
│   ├── Secure aggregation
│   ├── Differential privacy
│   └── Audit logging
└── Regulatory Layer
    ├── Compliance monitoring
    ├── Model explainability
    └── Audit trails
```

### Key Applications

| Application | Data Type | Fraud Ratio |
|-------------|-----------|-------------|
| Credit Card Fraud | Transactions | ~0.1% |
| Money Laundering | Transfers | ~0.5% |
| Identity Fraud | Applications | ~1% |
| Insurance Fraud | Claims | ~5% |

---

## Implementation Code

```python
#!/usr/bin/env python3
"""
Tutorial 147: Federated Learning Case Study - Finance

This module implements a federated learning system for financial
applications, focusing on cross-bank fraud detection.

Copyright 2026 Olaf Yunus Laitinen Imanov and Contributors
Released under EUPL 1.2
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import copy
import logging
from enum import Enum

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TransactionType(Enum):
    """Transaction classification."""
    LEGITIMATE = 0
    FRAUD = 1


@dataclass
class FinanceConfig:
    """Configuration for financial FL."""

    # FL parameters
    num_rounds: int = 50
    num_banks: int = 10
    banks_per_round: int = 8

    # Model parameters
    num_features: int = 30
    hidden_dim: int = 128
    num_classes: int = 2

    # Training parameters
    learning_rate: float = 0.001
    batch_size: int = 64
    local_epochs: int = 5

    # Fraud parameters
    fraud_ratio: float = 0.02  # 2% fraud rate

    # Privacy parameters
    differential_privacy: bool = True
    dp_epsilon: float = 1.0
    max_grad_norm: float = 1.0

    # Data parameters
    transactions_per_bank: int = 10000

    seed: int = 42


class TransactionDataset(Dataset):
    """Transaction dataset for fraud detection.

    Simulates realistic transaction data with:
    - Transaction amounts
    - Time features
    - Location features
    - Historical patterns
    """

    def __init__(
        self,
        bank_id: int,
        n: int = 10000,
        num_features: int = 30,
        fraud_ratio: float = 0.02,
        seed: int = 0
    ):
        np.random.seed(seed + bank_id)

        self.bank_id = bank_id
        self.n = n
        self.num_features = num_features

        # Generate transactions
        self.x, self.y = self._generate_transactions(fraud_ratio)

        # Compute class weights for balancing
        num_fraud = (self.y == 1).sum().item()
        num_legit = (self.y == 0).sum().item()
        self.class_weights = torch.tensor([
            1.0,
            num_legit / max(num_fraud, 1)
        ])

        logger.debug(
            f"Bank {bank_id}: {num_legit} legit, {num_fraud} fraud "
            f"({100*num_fraud/n:.2f}%)"
        )

    def _generate_transactions(
        self,
        fraud_ratio: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate synthetic transaction data."""

        features_list = []
        labels_list = []

        for _ in range(self.n):
            is_fraud = np.random.random() < fraud_ratio
            features = self._generate_transaction(is_fraud)

            features_list.append(features)
            labels_list.append(1 if is_fraud else 0)

        x = torch.tensor(np.array(features_list), dtype=torch.float32)
        y = torch.tensor(labels_list, dtype=torch.long)

        return x, y

    def _generate_transaction(self, is_fraud: bool) -> np.ndarray:
        """Generate a single transaction."""
        features = np.zeros(self.num_features)

        if is_fraud:
            # Fraudulent transaction patterns
            features[0] = np.random.lognormal(6, 1)  # Higher amounts
            features[1] = np.random.uniform(0, 6)    # Unusual hours
            features[2] = np.random.randint(0, 100)  # Random location
            features[3] = np.random.uniform(0, 0.1)  # New merchant
            features[4] = np.random.uniform(5, 10)   # Distance from usual

            # More fraud indicators
            for i in range(5, 15):
                features[i] = np.random.uniform(0.5, 1)
        else:
            # Legitimate transaction patterns
            features[0] = np.random.lognormal(4, 0.5)  # Normal amounts
            features[1] = np.random.triangular(8, 12, 20)  # Normal hours
            features[2] = np.random.randint(0, 10)     # Usual locations
            features[3] = np.random.uniform(0.5, 1)    # Known merchant
            features[4] = np.random.uniform(0, 2)      # Close to usual

            for i in range(5, 15):
                features[i] = np.random.uniform(0, 0.3)

        # Add noise and additional features
        for i in range(15, self.num_features):
            features[i] = np.random.randn()

        # Normalize
        features = (features - features.mean()) / (features.std() + 1e-8)

        return features

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx]


class FraudDetectionModel(nn.Module):
    """Neural network for fraud detection.

    Designed for highly imbalanced data with attention mechanism.
    """

    def __init__(self, config: FinanceConfig):
        super().__init__()

        self.config = config

        # Feature attention
        self.attention = nn.Sequential(
            nn.Linear(config.num_features, config.num_features // 2),
            nn.Tanh(),
            nn.Linear(config.num_features // 2, config.num_features),
            nn.Softmax(dim=-1)
        )

        # Main network
        self.encoder = nn.Sequential(
            nn.Linear(config.num_features, config.hidden_dim),
            nn.BatchNorm1d(config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.BatchNorm1d(config.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        self.classifier = nn.Linear(config.hidden_dim // 2, config.num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with attention."""
        attn_weights = self.attention(x)
        x_attended = x * attn_weights
        features = self.encoder(x_attended)
        return self.classifier(features)


class FraudMetrics:
    """Metrics for fraud detection evaluation."""

    @staticmethod
    def compute_metrics(
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> Dict[str, float]:
        """Compute fraud detection metrics."""

        pred_labels = predictions.argmax(dim=1)

        # Basic metrics
        accuracy = (pred_labels == targets).float().mean().item()

        # Confusion matrix elements
        tp = ((pred_labels == 1) & (targets == 1)).sum().item()
        fp = ((pred_labels == 1) & (targets == 0)).sum().item()
        tn = ((pred_labels == 0) & (targets == 0)).sum().item()
        fn = ((pred_labels == 0) & (targets == 1)).sum().item()

        # Precision, Recall, F1
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        # Specificity
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,  # Same as fraud detection rate
            "f1": f1,
            "specificity": specificity,
            "true_positives": tp,
            "false_positives": fp
        }


class BankClient:
    """FL client representing a bank."""

    def __init__(
        self,
        bank_id: int,
        dataset: TransactionDataset,
        config: FinanceConfig
    ):
        self.bank_id = bank_id
        self.dataset = dataset
        self.config = config

    def train(self, model: nn.Module) -> Dict[str, Any]:
        """Train on local transaction data."""

        local_model = copy.deepcopy(model)
        optimizer = torch.optim.Adam(
            local_model.parameters(),
            lr=self.config.learning_rate
        )

        # Use weighted sampler for imbalanced data
        sample_weights = torch.zeros(len(self.dataset))
        for i in range(len(self.dataset)):
            label = self.dataset.y[i].item()
            sample_weights[i] = self.dataset.class_weights[label]

        sampler = WeightedRandomSampler(
            sample_weights,
            num_samples=len(self.dataset),
            replacement=True
        )

        loader = DataLoader(
            self.dataset,
            batch_size=self.config.batch_size,
            sampler=sampler
        )

        local_model.train()
        total_loss = 0.0
        num_batches = 0

        for epoch in range(self.config.local_epochs):
            for x, y in loader:
                optimizer.zero_grad()

                output = local_model(x)
                loss = F.cross_entropy(
                    output, y,
                    weight=self.dataset.class_weights
                )

                loss.backward()

                # Gradient clipping (also for DP)
                torch.nn.utils.clip_grad_norm_(
                    local_model.parameters(),
                    self.config.max_grad_norm
                )

                # Add DP noise if enabled
                if self.config.differential_privacy:
                    for p in local_model.parameters():
                        if p.grad is not None:
                            noise = torch.randn_like(p.grad) * 0.01
                            p.grad.add_(noise)

                optimizer.step()

                total_loss += loss.item()
                num_batches += 1

        return {
            "state_dict": {
                k: v.cpu() for k, v in local_model.state_dict().items()
            },
            "num_samples": len(self.dataset),
            "avg_loss": total_loss / num_batches,
            "bank_id": self.bank_id
        }


class FinanceServer:
    """FL server for financial consortium."""

    def __init__(
        self,
        model: nn.Module,
        banks: List[BankClient],
        test_data: TransactionDataset,
        config: FinanceConfig
    ):
        self.model = model
        self.banks = banks
        self.test_data = test_data
        self.config = config
        self.history: List[Dict] = []

    def aggregate(self, updates: List[Dict[str, Any]]) -> None:
        """Aggregate bank updates."""
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

    def evaluate(self) -> Dict[str, float]:
        """Evaluate on test transactions."""
        self.model.eval()
        loader = DataLoader(self.test_data, batch_size=256)

        all_preds = []
        all_targets = []

        with torch.no_grad():
            for x, y in loader:
                output = self.model(x)
                all_preds.append(output)
                all_targets.append(y)

        predictions = torch.cat(all_preds)
        targets = torch.cat(all_targets)

        return FraudMetrics.compute_metrics(predictions, targets)

    def train(self) -> List[Dict]:
        """Run federated training."""
        logger.info(f"Starting financial FL with {len(self.banks)} banks")

        for round_num in range(self.config.num_rounds):
            # Select banks
            n = min(self.config.banks_per_round, len(self.banks))
            indices = np.random.choice(len(self.banks), n, replace=False)
            selected = [self.banks[i] for i in indices]

            # Collect updates
            updates = [b.train(self.model) for b in selected]

            # Aggregate
            self.aggregate(updates)

            # Evaluate
            metrics = self.evaluate()

            record = {
                "round": round_num,
                **metrics,
                "num_banks": len(selected),
                "avg_train_loss": np.mean([u["avg_loss"] for u in updates])
            }
            self.history.append(record)

            if (round_num + 1) % 10 == 0:
                logger.info(
                    f"Round {round_num + 1}: "
                    f"recall={metrics['recall']:.4f}, "
                    f"precision={metrics['precision']:.4f}, "
                    f"f1={metrics['f1']:.4f}"
                )

        return self.history


def main():
    """Main entry point."""
    print("=" * 60)
    print("Tutorial 147: FL Case Study - Finance")
    print("=" * 60)

    config = FinanceConfig()
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    # Create bank datasets
    banks = []
    for i in range(config.num_banks):
        # Vary fraud ratio slightly between banks
        bank_fraud_ratio = config.fraud_ratio * np.random.uniform(0.5, 1.5)

        dataset = TransactionDataset(
            bank_id=i,
            n=config.transactions_per_bank,
            num_features=config.num_features,
            fraud_ratio=bank_fraud_ratio,
            seed=config.seed
        )
        client = BankClient(i, dataset, config)
        banks.append(client)

    # Test data
    test_data = TransactionDataset(
        bank_id=999,
        n=5000,
        fraud_ratio=config.fraud_ratio,
        seed=999
    )

    # Model
    model = FraudDetectionModel(config)
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Train
    server = FinanceServer(model, banks, test_data, config)
    history = server.train()

    # Summary
    print("\n" + "=" * 60)
    print("Training Complete")
    print(f"Final Recall (Fraud Detection): {history[-1]['recall']:.4f}")
    print(f"Final Precision: {history[-1]['precision']:.4f}")
    print(f"Final F1 Score: {history[-1]['f1']:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
```

---

## Key Insights

### Financial FL Challenges

1. **Extreme Class Imbalance**: Fraud < 1% of transactions
2. **Regulatory Compliance**: GDPR, PCI-DSS requirements
3. **Competitive Sensitivity**: Banks reluctant to share data
4. **Real-time Requirements**: Fraud detection must be fast

### Best Practices

- Use weighted sampling for imbalanced data
- Apply attention for feature importance
- Implement differential privacy for compliance
- Track precision and recall, not just accuracy

---

## Exercises

1. **Exercise 1**: Add transaction sequence modeling
2. **Exercise 2**: Implement secure aggregation
3. **Exercise 3**: Add real-time fraud scoring
4. **Exercise 4**: Design explainable fraud alerts

---

## References

1. Yang, W., et al. (2019). FFD: A federated learning method for credit card fraud detection. In *BigData*.
2. Long, G., et al. (2020). Federated learning for open banking. In *IJCAI*.
3. Suzumura, T., et al. (2019). Towards federated graph learning for collaborative financial crimes detection. *arXiv*.

---

*Copyright 2026 Olaf Yunus Laitinen Imanov and Contributors. Released under EUPL 1.2.*
