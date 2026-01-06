# Tutorial 097: FL Differential Privacy

---

## Metadata

| Property | Value |
|----------|-------|
| **Tutorial ID** | 097 |
| **Title** | FL Differential Privacy |
| **Category** | Privacy |
| **Difficulty** | Expert |
| **Duration** | 120 minutes |
| **Prerequisites** | Tutorial 001-096 |
| **Author** | Unbitrium Contributors |
| **Last Updated** | January 2026 |

---

## Learning Objectives

By completing this tutorial, you will be able to:

1. **Understand** differential privacy fundamentals
2. **Implement** DP-SGD for FL
3. **Design** privacy budget management
4. **Analyze** privacy-utility trade-offs
5. **Deploy** private FL systems

---

## Background and Theory

### Differential Privacy

```
DP Components:
├── Privacy Parameters
│   ├── Epsilon (ε) - privacy loss
│   ├── Delta (δ) - failure probability
│   └── Sensitivity - max query change
├── Mechanisms
│   ├── Laplace mechanism
│   ├── Gaussian mechanism
│   └── Exponential mechanism
├── Composition
│   ├── Basic composition
│   ├── Advanced composition
│   └── Rényi DP
└── FL-Specific
    ├── Client-level DP
    ├── Record-level DP
    └── Local DP
```

---

## Implementation Code

```python
#!/usr/bin/env python3
"""
Tutorial 097: FL Differential Privacy

This module implements differential privacy for
federated learning with privacy guarantees.

Copyright 2026 Olaf Yunus Laitinen Imanov and Contributors
Released under EUPL 1.2
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass
from typing import Dict, List, Tuple
import copy
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DPConfig:
    """DP configuration."""

    num_rounds: int = 50
    num_clients: int = 20
    clients_per_round: int = 10

    input_dim: int = 32
    hidden_dim: int = 64
    num_classes: int = 10

    learning_rate: float = 0.01
    batch_size: int = 32
    local_epochs: int = 1

    # DP parameters
    target_epsilon: float = 8.0
    target_delta: float = 1e-5
    clip_norm: float = 1.0
    noise_multiplier: float = 1.0

    seed: int = 42


class PrivacyAccountant:
    """Track privacy budget."""

    def __init__(self, config: DPConfig):
        self.config = config
        self.epsilon_used = 0.0
        self.rounds = 0

    def compute_epsilon_per_round(self) -> float:
        """Compute epsilon per round."""
        q = self.config.clients_per_round / self.config.num_clients
        sigma = self.config.noise_multiplier

        # Simplified epsilon calculation
        epsilon = q * np.sqrt(2 * np.log(1.25 / self.config.target_delta)) / sigma
        return epsilon

    def step(self) -> float:
        """Record one round and return total epsilon."""
        self.rounds += 1
        eps_round = self.compute_epsilon_per_round()

        # Advanced composition
        self.epsilon_used = np.sqrt(self.rounds) * eps_round
        return self.epsilon_used

    def get_remaining(self) -> float:
        """Get remaining privacy budget."""
        return max(0, self.config.target_epsilon - self.epsilon_used)

    def is_depleted(self) -> bool:
        """Check if budget is exhausted."""
        return self.epsilon_used >= self.config.target_epsilon


class DPMechanism:
    """DP mechanisms for FL."""

    def __init__(self, config: DPConfig):
        self.config = config

    def clip_gradients(self, grads: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Clip gradients to bound sensitivity."""
        total_norm = sum((g ** 2).sum() for g in grads.values()) ** 0.5

        clip_factor = min(1.0, self.config.clip_norm / (total_norm + 1e-8))

        return {k: g * clip_factor for k, g in grads.items()}

    def add_gaussian_noise(self, grads: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Add calibrated Gaussian noise."""
        noise_std = self.config.clip_norm * self.config.noise_multiplier

        noisy_grads = {}
        for k, g in grads.items():
            noise = torch.randn_like(g) * noise_std
            noisy_grads[k] = g + noise

        return noisy_grads


class DPDataset(Dataset):
    def __init__(self, n: int = 200, dim: int = 32, classes: int = 10, seed: int = 0):
        np.random.seed(seed)
        self.x = torch.randn(n, dim, dtype=torch.float32)
        self.y = torch.randint(0, classes, (n,), dtype=torch.long)
        for i in range(n):
            self.x[i, self.y[i].item() % dim] += 2.0

    def __len__(self): return len(self.y)
    def __getitem__(self, idx): return self.x[idx], self.y[idx]


class DPModel(nn.Module):
    def __init__(self, config: DPConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.num_classes)
        )

    def forward(self, x): return self.net(x)


class DPClient:
    """Client with DP training."""

    def __init__(self, client_id: int, dataset: DPDataset, config: DPConfig):
        self.client_id = client_id
        self.dataset = dataset
        self.config = config

        self.dp_mechanism = DPMechanism(config)

    def train(self, model: nn.Module) -> Dict:
        local = copy.deepcopy(model)
        optimizer = torch.optim.SGD(local.parameters(), lr=self.config.learning_rate)
        loader = DataLoader(self.dataset, batch_size=self.config.batch_size, shuffle=True)

        local.train()
        total_loss = 0.0
        num_batches = 0

        for _ in range(self.config.local_epochs):
            for x, y in loader:
                optimizer.zero_grad()

                loss = F.cross_entropy(local(x), y)
                loss.backward()

                # Get gradients
                grads = {k: p.grad.clone() for k, p in local.named_parameters()}

                # Clip gradients
                clipped = self.dp_mechanism.clip_gradients(grads)

                # Apply clipped gradients
                for k, p in local.named_parameters():
                    p.grad = clipped[k]

                optimizer.step()

                total_loss += loss.item()
                num_batches += 1

        # Compute gradient diff for aggregation
        old_state = model.state_dict()
        new_state = local.state_dict()

        gradient = {k: new_state[k] - old_state[k] for k in old_state}

        # Clip and noise the gradient
        clipped = self.dp_mechanism.clip_gradients(gradient)
        noisy = self.dp_mechanism.add_gaussian_noise(clipped)

        return {
            "gradient": noisy,
            "num_samples": len(self.dataset),
            "avg_loss": total_loss / num_batches
        }


class DPServer:
    """Server with DP aggregation."""

    def __init__(self, model: nn.Module, clients: List[DPClient], test_data: DPDataset, config: DPConfig):
        self.model = model
        self.clients = clients
        self.test_data = test_data
        self.config = config

        self.accountant = PrivacyAccountant(config)
        self.history: List[Dict] = []

    def aggregate(self, updates: List[Dict]) -> None:
        n = len(updates)

        # Average gradients
        avg_gradient = {}
        for key in updates[0]["gradient"]:
            avg_gradient[key] = sum(u["gradient"][key] for u in updates) / n

        # Apply to model
        state = self.model.state_dict()
        for key in state:
            state[key] = state[key] + avg_gradient[key]
        self.model.load_state_dict(state)

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
        logger.info(f"Starting DP-FL (target ε={self.config.target_epsilon})")

        for round_num in range(self.config.num_rounds):
            if self.accountant.is_depleted():
                logger.info("Privacy budget exhausted!")
                break

            n = min(self.config.clients_per_round, len(self.clients))
            indices = np.random.choice(len(self.clients), n, replace=False)
            selected = [self.clients[i] for i in indices]

            updates = [c.train(self.model) for c in selected]
            self.aggregate(updates)

            epsilon = self.accountant.step()
            metrics = self.evaluate()

            record = {"round": round_num, "epsilon": epsilon, **metrics}
            self.history.append(record)

            if (round_num + 1) % 10 == 0:
                logger.info(f"Round {round_num + 1}: acc={metrics['accuracy']:.4f}, ε={epsilon:.2f}")

        return self.history


def main():
    print("=" * 60)
    print("Tutorial 097: FL Differential Privacy")
    print("=" * 60)

    config = DPConfig()
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    clients = [DPClient(i, DPDataset(seed=config.seed + i), config) for i in range(config.num_clients)]
    test_data = DPDataset(seed=999)
    model = DPModel(config)

    server = DPServer(model, clients, test_data, config)
    history = server.train()

    print("\n" + "=" * 60)
    print("DP Training Complete")
    print(f"Final accuracy: {history[-1]['accuracy']:.4f}")
    print(f"Final epsilon: {history[-1]['epsilon']:.4f}")
    print(f"Rounds completed: {len(history)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
```

---

## Key Insights

### DP Best Practices

1. **Budget management**: Track cumulative epsilon
2. **Noise calibration**: Match to sensitivity
3. **Subsampling**: Amplify privacy via sampling
4. **Early stopping**: Stop when budget depleted

---

## Exercises

1. **Exercise 1**: Implement Rényi DP accounting
2. **Exercise 2**: Add local DP
3. **Exercise 3**: Design adaptive clipping
4. **Exercise 4**: Compare composition methods

---

## References

1. Dwork, C., & Roth, A. (2014). The algorithmic foundations of differential privacy.
2. McMahan, H.B., et al. (2018). Learning differentially private recurrent language models. In *ICLR*.
3. Abadi, M., et al. (2016). Deep learning with differential privacy. In *CCS*.

---

*Copyright 2026 Olaf Yunus Laitinen Imanov and Contributors. Released under EUPL 1.2.*
