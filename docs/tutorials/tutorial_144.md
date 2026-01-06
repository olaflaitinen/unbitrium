# Tutorial 144: FL Open Problems

---

## Metadata

| Property | Value |
|----------|-------|
| **Tutorial ID** | 144 |
| **Title** | Federated Learning Open Problems |
| **Category** | Research |
| **Difficulty** | Expert |
| **Duration** | 120 minutes |
| **Prerequisites** | Tutorial 001-143 |
| **Author** | Unbitrium Contributors |
| **Last Updated** | January 2026 |

---

## Learning Objectives

By completing this tutorial, you will be able to:

1. **Understand** major open problems in FL
2. **Analyze** current research challenges
3. **Identify** directions for contribution
4. **Evaluate** proposed solutions
5. **Formulate** research questions

---

## Prerequisites

Before starting this tutorial, ensure you have:

- Completed tutorials 001-143
- Strong FL fundamentals
- Research-level understanding
- Familiarity with current literature

---

## Background and Theory

### Why Open Problems Matter

FL is an active research area with many unsolved challenges.
Understanding these helps researchers and practitioners:
- Identify contribution opportunities
- Understand system limitations
- Anticipate future developments
- Design robust systems

### Open Problem Categories

```
FL Open Problems:
├── Statistical Challenges
│   ├── Non-IID data handling
│   ├── Label distribution shift
│   └── Feature heterogeneity
├── Systems Challenges  
│   ├── Scalability limits
│   ├── Async aggregation
│   └── Fault tolerance
├── Security & Privacy
│   ├── Byzantine attacks
│   ├── Privacy leakage
│   └── Inference attacks
├── Optimization
│   ├── Convergence guarantees
│   ├── Client drift
│   └── Communication efficiency
└── Practical Deployment
    ├── Model debugging
    ├── Fairness
    └── Incentive design
```

---

## Implementation Code

```python
#!/usr/bin/env python3
"""
Tutorial 144: Federated Learning Open Problems

This module demonstrates key open problems in FL through
experimental implementations and analysis.

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
from enum import Enum

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class OpenProblemConfig:
    """Configuration for open problem experiments."""

    num_rounds: int = 50
    num_clients: int = 20
    clients_per_round: int = 10

    input_dim: int = 32
    hidden_dim: int = 64
    num_classes: int = 10

    learning_rate: float = 0.01
    batch_size: int = 32
    local_epochs: int = 5

    seed: int = 42


class OpenProblemDataset(Dataset):
    """Dataset for open problem experiments."""

    def __init__(
        self,
        client_id: int,
        n: int = 200,
        dim: int = 32,
        classes: int = 10,
        seed: int = 0
    ):
        np.random.seed(seed + client_id)

        self.x = torch.randn(n, dim, dtype=torch.float32)
        self.y = torch.randint(0, classes, (n,), dtype=torch.long)

        for i in range(n):
            self.x[i, self.y[i].item() % dim] += 2.0

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx]


class OpenProblemModel(nn.Module):
    """Model for experiments."""

    def __init__(self, config: OpenProblemConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# =============================================================================
# Open Problem 1: Non-IID Data
# =============================================================================

class NonIIDExperiment:
    """Experiment demonstrating non-IID challenges."""

    def __init__(self, config: OpenProblemConfig):
        self.config = config

    def create_non_iid_data(
        self,
        num_clients: int,
        alpha: float = 0.1
    ) -> List[OpenProblemDataset]:
        """Create highly non-IID data using Dirichlet."""
        datasets = []

        for i in range(num_clients):
            # Very skewed class distribution
            probs = np.random.dirichlet([alpha] * self.config.num_classes)

            dataset = OpenProblemDataset(
                client_id=i,
                dim=self.config.input_dim,
                classes=self.config.num_classes,
                seed=self.config.seed
            )

            # Resample with skew
            new_labels = np.random.choice(
                self.config.num_classes,
                size=len(dataset),
                p=probs
            )
            dataset.y = torch.tensor(new_labels, dtype=torch.long)

            datasets.append(dataset)

        return datasets

    def measure_divergence(
        self,
        client_updates: List[Dict]
    ) -> Dict[str, float]:
        """Measure update divergence across clients."""
        if len(client_updates) < 2:
            return {"divergence": 0.0}

        # Compute pairwise divergence
        divergences = []
        for i in range(len(client_updates)):
            for j in range(i + 1, len(client_updates)):
                div = 0.0
                for key in client_updates[i]["state_dict"]:
                    diff = (
                        client_updates[i]["state_dict"][key] -
                        client_updates[j]["state_dict"][key]
                    )
                    div += (diff ** 2).sum().item()
                divergences.append(div ** 0.5)

        return {
            "mean_divergence": np.mean(divergences),
            "max_divergence": np.max(divergences),
            "std_divergence": np.std(divergences)
        }


# =============================================================================
# Open Problem 2: Byzantine Robustness
# =============================================================================

class ByzantineAttack(Enum):
    """Types of Byzantine attacks."""
    NONE = "none"
    RANDOM = "random"
    SIGN_FLIP = "sign_flip"
    LABEL_FLIP = "label_flip"


class ByzantineExperiment:
    """Experiment for Byzantine robustness."""

    def __init__(
        self,
        config: OpenProblemConfig,
        attack_type: ByzantineAttack = ByzantineAttack.SIGN_FLIP,
        byzantine_ratio: float = 0.2
    ):
        self.config = config
        self.attack_type = attack_type
        self.byzantine_ratio = byzantine_ratio

    def apply_attack(
        self,
        update: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Apply Byzantine attack to update."""

        if self.attack_type == ByzantineAttack.NONE:
            return update

        attacked = {}

        for key, value in update.items():
            if self.attack_type == ByzantineAttack.RANDOM:
                attacked[key] = torch.randn_like(value) * value.std() * 10

            elif self.attack_type == ByzantineAttack.SIGN_FLIP:
                attacked[key] = -value * 5

            else:
                attacked[key] = value

        return attacked

    def median_aggregation(
        self,
        updates: List[Dict]
    ) -> Dict[str, torch.Tensor]:
        """Coordinate-wise median for robustness."""
        new_state = {}

        for key in updates[0]["state_dict"]:
            stacked = torch.stack([
                u["state_dict"][key] for u in updates
            ])
            new_state[key] = torch.median(stacked, dim=0).values

        return new_state

    def trimmed_mean(
        self,
        updates: List[Dict],
        trim_ratio: float = 0.2
    ) -> Dict[str, torch.Tensor]:
        """Trimmed mean for robustness."""
        new_state = {}
        k = int(len(updates) * trim_ratio)

        for key in updates[0]["state_dict"]:
            stacked = torch.stack([
                u["state_dict"][key] for u in updates
            ])

            # Sort and trim
            sorted_vals, _ = torch.sort(stacked, dim=0)
            trimmed = sorted_vals[k:-k] if k > 0 else sorted_vals
            new_state[key] = trimmed.mean(dim=0)

        return new_state


# =============================================================================
# Open Problem 3: Privacy Leakage
# =============================================================================

class PrivacyExperiment:
    """Experiment analyzing privacy leakage."""

    def __init__(self, config: OpenProblemConfig):
        self.config = config

    def gradient_leakage_attack(
        self,
        original_x: torch.Tensor,
        original_y: torch.Tensor,
        model: nn.Module,
        gradient: Dict[str, torch.Tensor],
        num_iterations: int = 1000
    ) -> Tuple[torch.Tensor, float]:
        """
        Attempt to reconstruct data from gradients.

        This demonstrates the Deep Leakage from Gradients attack.
        """
        # Initialize dummy data
        dummy_x = torch.randn_like(original_x, requires_grad=True)
        dummy_y = original_y.clone()

        optimizer = torch.optim.LBFGS([dummy_x])

        def closure():
            optimizer.zero_grad()

            # Compute dummy gradient
            output = model(dummy_x)
            loss = F.cross_entropy(output, dummy_y)
            dummy_grad = torch.autograd.grad(loss, model.parameters(), create_graph=True)

            # Match gradients
            grad_diff = 0.0
            for (name, param), dg in zip(model.named_parameters(), dummy_grad):
                if name in gradient:
                    grad_diff += ((gradient[name] - dg) ** 2).sum()

            grad_diff.backward()
            return grad_diff

        for _ in range(num_iterations):
            optimizer.step(closure)

        # Compute reconstruction error
        error = F.mse_loss(dummy_x.detach(), original_x).item()

        return dummy_x.detach(), error

    def add_differential_privacy(
        self,
        gradient: Dict[str, torch.Tensor],
        clip_norm: float = 1.0,
        noise_scale: float = 0.1
    ) -> Dict[str, torch.Tensor]:
        """Add DP noise to gradients."""
        noisy_gradient = {}

        # Clip
        total_norm = sum(
            (g ** 2).sum().item() for g in gradient.values()
        ) ** 0.5
        clip_factor = min(1.0, clip_norm / (total_norm + 1e-8))

        for name, grad in gradient.items():
            clipped = grad * clip_factor
            noise = torch.randn_like(clipped) * noise_scale * clip_norm
            noisy_gradient[name] = clipped + noise

        return noisy_gradient


# =============================================================================
# Open Problem 4: Client Drift
# =============================================================================

class ClientDriftExperiment:
    """Experiment analyzing client drift."""

    def __init__(self, config: OpenProblemConfig):
        self.config = config

    def measure_drift(
        self,
        initial_model: nn.Module,
        final_model: nn.Module
    ) -> float:
        """Measure model drift during local training."""
        drift = 0.0

        for (_, p1), (_, p2) in zip(
            initial_model.named_parameters(),
            final_model.named_parameters()
        ):
            drift += ((p1 - p2) ** 2).sum().item()

        return drift ** 0.5

    def proximal_regularization(
        self,
        local_model: nn.Module,
        global_model: nn.Module,
        loss: torch.Tensor,
        mu: float = 0.01
    ) -> torch.Tensor:
        """Add proximal term to reduce drift."""
        prox_term = 0.0

        for (_, local_p), (_, global_p) in zip(
            local_model.named_parameters(),
            global_model.named_parameters()
        ):
            prox_term += ((local_p - global_p.detach()) ** 2).sum()

        return loss + (mu / 2) * prox_term


# =============================================================================
# Open Problem 5: Fairness
# =============================================================================

class FairnessExperiment:
    """Experiment analyzing fairness across clients."""

    def __init__(self, config: OpenProblemConfig):
        self.config = config

    def measure_performance_variance(
        self,
        model: nn.Module,
        client_datasets: List[OpenProblemDataset]
    ) -> Dict[str, float]:
        """Measure performance variance across clients."""
        model.eval()
        accuracies = []

        for dataset in client_datasets:
            loader = DataLoader(dataset, batch_size=64)
            correct = 0
            total = 0

            with torch.no_grad():
                for x, y in loader:
                    pred = model(x).argmax(dim=1)
                    correct += (pred == y).sum().item()
                    total += len(y)

            accuracies.append(correct / total)

        return {
            "mean_accuracy": np.mean(accuracies),
            "std_accuracy": np.std(accuracies),
            "min_accuracy": np.min(accuracies),
            "max_accuracy": np.max(accuracies),
            "accuracy_gap": np.max(accuracies) - np.min(accuracies)
        }

    def agnostic_fair_aggregate(
        self,
        updates: List[Dict],
        accuracies: List[float]
    ) -> Dict[str, torch.Tensor]:
        """Agnostic fair aggregation - prioritize low-performing clients."""
        # Invert accuracies for weighting
        weights = 1.0 / (np.array(accuracies) + 0.1)
        weights = weights / weights.sum()

        new_state = {}
        for key in updates[0]["state_dict"]:
            new_state[key] = sum(
                w * u["state_dict"][key].float()
                for w, u in zip(weights, updates)
            )

        return new_state


class OpenProblemClient:
    """Client for open problem experiments."""

    def __init__(
        self,
        client_id: int,
        dataset: OpenProblemDataset,
        config: OpenProblemConfig
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
            "avg_loss": total_loss / num_batches
        }


def main():
    """Main entry point."""
    print("=" * 60)
    print("Tutorial 144: FL Open Problems")
    print("=" * 60)

    config = OpenProblemConfig()
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    # Initialize experiments
    non_iid_exp = NonIIDExperiment(config)
    byzantine_exp = ByzantineExperiment(config)
    privacy_exp = PrivacyExperiment(config)
    drift_exp = ClientDriftExperiment(config)
    fairness_exp = FairnessExperiment(config)

    # Create datasets
    datasets = non_iid_exp.create_non_iid_data(config.num_clients, alpha=0.1)

    # Measure fairness
    model = OpenProblemModel(config)
    fairness_metrics = fairness_exp.measure_performance_variance(model, datasets)

    print("\nFairness Analysis (with random model):")
    for k, v in fairness_metrics.items():
        print(f"  {k}: {v:.4f}")

    # Create clients and train
    clients = [OpenProblemClient(i, d, config) for i, d in enumerate(datasets)]
    updates = [c.train(model) for c in clients[:5]]

    # Measure divergence
    divergence = non_iid_exp.measure_divergence(updates)
    print("\nUpdate Divergence:")
    for k, v in divergence.items():
        print(f"  {k}: {v:.4f}")

    print("\n" + "=" * 60)
    print("Open Problem Experiments Complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
```

---

## Key Open Problems Summary

| Problem | Challenge | Current Solutions | Limitations |
|---------|-----------|-------------------|-------------|
| Non-IID Data | Performance degradation | FedProx, SCAFFOLD | Added complexity |
| Byzantine Attacks | Malicious clients | Median, Krum | Reduced accuracy |
| Privacy Leakage | Data reconstruction | DP, SecAgg | Utility loss |
| Client Drift | Local optima divergence | Proximal terms | Slower convergence |
| Fairness | Performance disparity | Fair FL | Trade-offs |

---

## Exercises

1. **Exercise 1**: Implement gradient leakage attack
2. **Exercise 2**: Design new Byzantine defense
3. **Exercise 3**: Measure privacy-utility tradeoff
4. **Exercise 4**: Propose fairness metric

---

## References

1. Kairouz, P., et al. (2021). Advances and open problems in FL. *FnTML*.
2. Zhu, L., et al. (2019). Deep leakage from gradients. In *NeurIPS*.
3. Li, T., et al. (2020). Fair resource allocation in FL. In *ICLR*.

---

*Copyright 2026 Olaf Yunus Laitinen Imanov and Contributors. Released under EUPL 1.2.*
