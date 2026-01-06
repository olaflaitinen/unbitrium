# Tutorial 095: FL Privacy Attacks

---

## Metadata

| Property | Value |
|----------|-------|
| **Tutorial ID** | 095 |
| **Title** | Federated Learning Privacy Attacks |
| **Category** | Security |
| **Difficulty** | Expert |
| **Duration** | 120 minutes |
| **Prerequisites** | Tutorial 001-094 |
| **Author** | Unbitrium Contributors |
| **Last Updated** | January 2026 |

---

## Learning Objectives

By completing this tutorial, you will be able to:

1. **Understand** privacy attack vectors
2. **Implement** gradient inversion attacks
3. **Analyze** membership inference
4. **Design** attack defenses
5. **Evaluate** privacy risks

---

## Prerequisites

Before starting this tutorial, ensure you have:

- Completed tutorials 001-094
- Understanding of FL fundamentals
- Knowledge of privacy attacks
- Familiarity with optimization

---

## Background and Theory

### Privacy Attack Taxonomy

```
FL Privacy Attacks:
├── Gradient Inversion
│   ├── Image reconstruction
│   ├── Text recovery
│   └── Optimization-based
├── Membership Inference
│   ├── Is record in training?
│   ├── Shadow models
│   └── Loss-based detection
├── Model Inversion
│   ├── Reconstruct features
│   ├── Class representatives
│   └── GAN-based
└── Property Inference
    ├── Data distribution
    ├── Sensitive attributes
    └── Training details
```

---

## Implementation Code

```python
#!/usr/bin/env python3
"""
Tutorial 095: FL Privacy Attacks

This module implements privacy attacks against
federated learning for security analysis.

Copyright 2026 Olaf Yunus Laitinen Imanov and Contributors
Released under EUPL 1.2
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import copy
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class AttackConfig:
    """Privacy attack configuration."""

    num_rounds: int = 20
    num_clients: int = 10
    clients_per_round: int = 5

    input_dim: int = 32
    hidden_dim: int = 64
    num_classes: int = 10

    learning_rate: float = 0.01
    batch_size: int = 32
    local_epochs: int = 1

    # Attack params
    attack_iterations: int = 1000
    attack_lr: float = 0.1

    seed: int = 42


class AttackModel(nn.Module):
    def __init__(self, config: AttackConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.num_classes)
        )

    def forward(self, x): return self.net(x)


class AttackDataset(Dataset):
    def __init__(self, n: int = 200, dim: int = 32, classes: int = 10, seed: int = 0):
        np.random.seed(seed)
        self.x = torch.randn(n, dim, dtype=torch.float32)
        self.y = torch.randint(0, classes, (n,), dtype=torch.long)
        for i in range(n):
            self.x[i, self.y[i].item() % dim] += 2.0

    def __len__(self): return len(self.y)
    def __getitem__(self, idx): return self.x[idx], self.y[idx]


class GradientInversionAttack:
    """Reconstruct data from gradients."""

    def __init__(self, config: AttackConfig):
        self.config = config

    def attack(
        self,
        model: nn.Module,
        gradients: Dict[str, torch.Tensor],
        true_label: Optional[int] = None
    ) -> Tuple[torch.Tensor, Optional[int]]:
        """Attempt to reconstruct input from gradients."""
        # Initialize dummy data
        dummy_data = torch.randn(1, self.config.input_dim, requires_grad=True)

        # Initialize dummy label if not known
        if true_label is not None:
            dummy_label = torch.tensor([true_label])
        else:
            dummy_label = torch.randint(0, self.config.num_classes, (1,))

        optimizer = torch.optim.LBFGS([dummy_data], lr=self.config.attack_lr)

        for iteration in range(self.config.attack_iterations):
            def closure():
                optimizer.zero_grad()

                # Compute gradients for dummy data
                model.zero_grad()
                output = model(dummy_data)
                loss = F.cross_entropy(output, dummy_label)
                loss.backward()

                dummy_grads = {k: p.grad.clone() for k, p in model.named_parameters()}

                # Match gradients
                grad_diff = 0.0
                for key in gradients:
                    grad_diff += ((dummy_grads[key] - gradients[key]) ** 2).sum()

                grad_diff.backward()
                return grad_diff

            optimizer.step(closure)

        return dummy_data.detach(), dummy_label.item()


class MembershipInferenceAttack:
    """Determine if record was in training."""

    def __init__(self, config: AttackConfig):
        self.config = config
        self.shadow_models: List[nn.Module] = []
        self.threshold = 0.0

    def train_shadow_models(self, num_shadows: int = 3) -> None:
        """Train shadow models for attack."""
        for i in range(num_shadows):
            shadow = AttackModel(self.config)
            shadow_data = AttackDataset(seed=self.config.seed + 1000 + i)

            optimizer = torch.optim.Adam(shadow.parameters(), lr=self.config.learning_rate)
            loader = DataLoader(shadow_data, batch_size=self.config.batch_size, shuffle=True)

            shadow.train()
            for _ in range(5):
                for x, y in loader:
                    optimizer.zero_grad()
                    loss = F.cross_entropy(shadow(x), y)
                    loss.backward()
                    optimizer.step()

            self.shadow_models.append(shadow)

        # Compute threshold
        losses_in, losses_out = [], []

        for shadow in self.shadow_models:
            shadow.eval()
            in_data = AttackDataset(n=50, seed=self.config.seed + 2000)
            out_data = AttackDataset(n=50, seed=self.config.seed + 3000)

            with torch.no_grad():
                for x, y in DataLoader(in_data, batch_size=1):
                    loss = F.cross_entropy(shadow(x), y)
                    losses_in.append(loss.item())

                for x, y in DataLoader(out_data, batch_size=1):
                    loss = F.cross_entropy(shadow(x), y)
                    losses_out.append(loss.item())

        self.threshold = (np.mean(losses_in) + np.mean(losses_out)) / 2

    def attack(self, model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> bool:
        """Determine membership."""
        model.eval()
        with torch.no_grad():
            loss = F.cross_entropy(model(x.unsqueeze(0)), y.unsqueeze(0))

        return loss.item() < self.threshold


class PrivacyAttacker:
    """Coordinate privacy attacks."""

    def __init__(self, config: AttackConfig):
        self.config = config
        self.gradient_attack = GradientInversionAttack(config)
        self.membership_attack = MembershipInferenceAttack(config)

    def analyze_gradient_attack(
        self,
        model: nn.Module,
        true_data: torch.Tensor,
        true_label: int
    ) -> Dict[str, float]:
        """Analyze gradient inversion effectiveness."""
        # Get true gradients
        model.zero_grad()
        output = model(true_data.unsqueeze(0))
        loss = F.cross_entropy(output, torch.tensor([true_label]))
        loss.backward()

        true_grads = {k: p.grad.clone() for k, p in model.named_parameters()}

        # Attempt reconstruction
        reconstructed, pred_label = self.gradient_attack.attack(
            model, true_grads, true_label
        )

        # Measure quality
        mse = ((reconstructed - true_data) ** 2).mean().item()
        cosine_sim = F.cosine_similarity(
            reconstructed.flatten().unsqueeze(0),
            true_data.flatten().unsqueeze(0)
        ).item()

        return {
            "mse": mse,
            "cosine_similarity": cosine_sim,
            "label_correct": pred_label == true_label
        }

    def analyze_membership_attack(
        self,
        model: nn.Module,
        in_samples: List[Tuple[torch.Tensor, torch.Tensor]],
        out_samples: List[Tuple[torch.Tensor, torch.Tensor]]
    ) -> Dict[str, float]:
        """Analyze membership inference effectiveness."""
        self.membership_attack.train_shadow_models()

        tp, fp, tn, fn = 0, 0, 0, 0

        for x, y in in_samples:
            if self.membership_attack.attack(model, x, y):
                tp += 1
            else:
                fn += 1

        for x, y in out_samples:
            if self.membership_attack.attack(model, x, y):
                fp += 1
            else:
                tn += 1

        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall
        }


def main():
    print("=" * 60)
    print("Tutorial 095: FL Privacy Attacks")
    print("=" * 60)

    config = AttackConfig()
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    # Train a model
    model = AttackModel(config)
    train_data = AttackDataset(seed=config.seed)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    loader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True)

    model.train()
    for _ in range(10):
        for x, y in loader:
            optimizer.zero_grad()
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            optimizer.step()

    # Analyze attacks
    attacker = PrivacyAttacker(config)

    # Gradient inversion
    test_x, test_y = train_data[0]
    grad_results = attacker.analyze_gradient_attack(model, test_x, test_y.item())

    print("\nGradient Inversion Attack:")
    print(f"  MSE: {grad_results['mse']:.4f}")
    print(f"  Cosine Similarity: {grad_results['cosine_similarity']:.4f}")

    # Membership inference
    in_samples = [(train_data[i][0], train_data[i][1]) for i in range(20)]
    out_samples = [(x, y) for x, y in DataLoader(AttackDataset(seed=9999), batch_size=1)][:20]

    membership_results = attacker.analyze_membership_attack(model, in_samples, out_samples)

    print("\nMembership Inference Attack:")
    print(f"  Accuracy: {membership_results['accuracy']:.4f}")
    print(f"  Precision: {membership_results['precision']:.4f}")
    print(f"  Recall: {membership_results['recall']:.4f}")

    print("=" * 60)


if __name__ == "__main__":
    main()
```

---

## Key Insights

### Attack Defense Strategies

1. **Differential privacy**: Add noise
2. **Secure aggregation**: Hide individual updates
3. **Gradient clipping**: Limit information
4. **Regularization**: Reduce memorization

---

## Exercises

1. **Exercise 1**: Implement model inversion
2. **Exercise 2**: Add property inference
3. **Exercise 3**: Design defense evaluation
4. **Exercise 4**: Combine attack types

---

## References

1. Zhu, L., et al. (2019). Deep leakage from gradients. In *NeurIPS*.
2. Shokri, R., et al. (2017). Membership inference attacks. In *S&P*.
3. Fredrikson, M., et al. (2015). Model inversion attacks. In *CCS*.

---

*Copyright 2026 Olaf Yunus Laitinen Imanov and Contributors. Released under EUPL 1.2.*
