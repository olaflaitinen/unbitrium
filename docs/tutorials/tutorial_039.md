# Tutorial 039: FL Data Augmentation

---

## Metadata

| Property | Value |
|----------|-------|
| **Tutorial ID** | 039 |
| **Title** | Federated Learning Data Augmentation |
| **Category** | Data Techniques |
| **Difficulty** | Intermediate |
| **Duration** | 90 minutes |
| **Prerequisites** | Tutorial 001-038 |
| **Author** | Unbitrium Contributors |
| **Last Updated** | January 2026 |

---

## Learning Objectives

By completing this tutorial, you will be able to:

1. **Understand** augmentation in FL contexts
2. **Implement** federated augmentation strategies
3. **Design** privacy-preserving augmentation
4. **Analyze** augmentation impact on FL
5. **Deploy** efficient augmentation pipelines

---

## Prerequisites

Before starting this tutorial, ensure you have:

- Completed tutorials 001-038
- Understanding of FL fundamentals
- Knowledge of data augmentation
- Familiarity with image transformations

---

## Background and Theory

### Why Augmentation in FL?

Data augmentation is crucial in FL because:
- Client data is often limited
- Non-IID distributions need balancing
- Model generalization requires diversity
- Privacy prevents data sharing

### Augmentation Strategies

```
FL Augmentation:
├── Local Augmentation
│   ├── Standard transforms
│   ├── MixUp / CutMix
│   └── AutoAugment
├── Federated Augmentation
│   ├── Shared policies
│   ├── Learned augmentations
│   └── Distribution matching
└── Synthetic Data
    ├── GAN-based generation
    ├── Diffusion models
    └── Feature mixup
```

---

## Implementation Code

```python
#!/usr/bin/env python3
"""
Tutorial 039: FL Data Augmentation

This module implements data augmentation strategies
for federated learning.

Copyright 2026 Olaf Yunus Laitinen Imanov and Contributors
Released under EUPL 1.2
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable
import copy
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class AugConfig:
    """Augmentation configuration."""

    num_rounds: int = 50
    num_clients: int = 10
    clients_per_round: int = 5

    input_dim: int = 32
    hidden_dim: int = 64
    num_classes: int = 10

    learning_rate: float = 0.01
    batch_size: int = 32
    local_epochs: int = 3

    # Augmentation params
    enable_augmentation: bool = True
    mixup_alpha: float = 0.2
    noise_std: float = 0.1

    seed: int = 42


class Augmentor:
    """Data augmentation for FL."""

    def __init__(self, config: AugConfig):
        self.config = config
        self.rng = np.random.RandomState(config.seed)

    def add_noise(self, x: torch.Tensor) -> torch.Tensor:
        """Add Gaussian noise."""
        noise = torch.randn_like(x) * self.config.noise_std
        return x + noise

    def mixup(
        self,
        x: torch.Tensor,
        y: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """MixUp augmentation."""
        alpha = self.config.mixup_alpha
        lam = np.random.beta(alpha, alpha) if alpha > 0 else 1

        batch_size = x.size(0)
        index = torch.randperm(batch_size)

        mixed_x = lam * x + (1 - lam) * x[index]
        y_a, y_b = y, y[index]

        return mixed_x, y_a, y_b, lam

    def random_erasing(
        self,
        x: torch.Tensor,
        probability: float = 0.5
    ) -> torch.Tensor:
        """Random erasing augmentation."""
        if self.rng.random() > probability:
            return x

        batch, dim = x.shape
        erase_dim = max(1, int(dim * 0.2))
        start = self.rng.randint(0, dim - erase_dim)

        x_aug = x.clone()
        x_aug[:, start:start + erase_dim] = 0

        return x_aug

    def augment_batch(
        self,
        x: torch.Tensor,
        y: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply all augmentations."""
        x_aug = self.add_noise(x)
        x_aug = self.random_erasing(x_aug)

        return x_aug, y


class AugDataset(Dataset):
    """Dataset with augmentation."""

    def __init__(
        self,
        n: int = 200,
        dim: int = 32,
        classes: int = 10,
        seed: int = 0,
        augmentor: Optional[Augmentor] = None
    ):
        np.random.seed(seed)
        self.x = torch.randn(n, dim, dtype=torch.float32)
        self.y = torch.randint(0, classes, (n,), dtype=torch.long)

        for i in range(n):
            self.x[i, self.y[i].item() % dim] += 2.0

        self.augmentor = augmentor

    def __len__(self): return len(self.y)

    def __getitem__(self, idx):
        x, y = self.x[idx], self.y[idx]

        if self.augmentor:
            x = self.augmentor.add_noise(x.unsqueeze(0)).squeeze(0)

        return x, y


class AugModel(nn.Module):
    def __init__(self, config: AugConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(config.hidden_dim, config.num_classes)
        )

    def forward(self, x): return self.net(x)


class AugClient:
    """Client with augmentation."""

    def __init__(
        self,
        client_id: int,
        dataset: AugDataset,
        config: AugConfig
    ):
        self.client_id = client_id
        self.dataset = dataset
        self.config = config
        self.augmentor = Augmentor(config) if config.enable_augmentation else None

    def train(self, model: nn.Module) -> Dict:
        local = copy.deepcopy(model)
        optimizer = torch.optim.Adam(local.parameters(), lr=self.config.learning_rate)
        loader = DataLoader(self.dataset, batch_size=self.config.batch_size, shuffle=True)

        local.train()
        total_loss, num_batches = 0.0, 0

        for _ in range(self.config.local_epochs):
            for x, y in loader:
                optimizer.zero_grad()

                if self.augmentor and np.random.random() < 0.5:
                    # Apply MixUp
                    x_mixed, y_a, y_b, lam = self.augmentor.mixup(x, y)
                    output = local(x_mixed)
                    loss = lam * F.cross_entropy(output, y_a) + \
                           (1 - lam) * F.cross_entropy(output, y_b)
                else:
                    # Standard training
                    if self.augmentor:
                        x, y = self.augmentor.augment_batch(x, y)
                    output = local(x)
                    loss = F.cross_entropy(output, y)

                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                num_batches += 1

        return {
            "state_dict": {k: v.cpu() for k, v in local.state_dict().items()},
            "num_samples": len(self.dataset),
            "avg_loss": total_loss / num_batches
        }


class AugServer:
    """Server for FL with augmentation."""

    def __init__(
        self,
        model: nn.Module,
        clients: List[AugClient],
        test_data: AugDataset,
        config: AugConfig
    ):
        self.model = model
        self.clients = clients
        self.test_data = test_data
        self.config = config
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
                pred = self.model(x).argmax(dim=1)
                correct += (pred == y).sum().item()
                total += len(y)
        return {"accuracy": correct / total}

    def train(self) -> List[Dict]:
        aug_status = "enabled" if self.config.enable_augmentation else "disabled"
        logger.info(f"Starting FL with augmentation {aug_status}")

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
    print("Tutorial 039: FL Data Augmentation")
    print("=" * 60)

    config = AugConfig()
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    # Compare with/without augmentation
    results = {}

    for enable_aug in [False, True]:
        config.enable_augmentation = enable_aug

        clients = [
            AugClient(i, AugDataset(n=100, seed=config.seed + i), config)
            for i in range(config.num_clients)
        ]
        test_data = AugDataset(n=300, seed=999)
        model = AugModel(config)

        server = AugServer(model, clients, test_data, config)
        history = server.train()

        results[f"aug_{enable_aug}"] = history[-1]["accuracy"]

    print("\n" + "=" * 60)
    print("Augmentation Impact")
    print(f"  Without augmentation: {results['aug_False']:.4f}")
    print(f"  With augmentation: {results['aug_True']:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
```

---

## Key Insights

### Augmentation Best Practices

1. **Local augmentation**: Preserve privacy
2. **MixUp**: Improve generalization
3. **Consistent policy**: Shared across clients
4. **Validate impact**: A/B test thoroughly

---

## Exercises

1. **Exercise 1**: Implement CutMix
2. **Exercise 2**: Add AutoAugment
3. **Exercise 3**: Design federated augmentation search
4. **Exercise 4**: Add privacy-preserving synthetic data

---

## References

1. Zhang, H., et al. (2018). mixup: Beyond empirical risk minimization. In *ICLR*.
2. Cubuk, E.D., et al. (2019). AutoAugment: Learning augmentation policies from data. In *CVPR*.
3. DeVries, T., & Taylor, G.W. (2017). Improved regularization with cutout. *arXiv*.

---

*Copyright 2026 Olaf Yunus Laitinen Imanov and Contributors. Released under EUPL 1.2.*
