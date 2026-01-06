# Tutorial 123: FL AutoML

---

## Metadata

| Property | Value |
|----------|-------|
| **Tutorial ID** | 123 |
| **Title** | Federated Learning AutoML |
| **Category** | Automation |
| **Difficulty** | Expert |
| **Duration** | 120 minutes |
| **Prerequisites** | Tutorial 001-122 |
| **Author** | Unbitrium Contributors |
| **Last Updated** | January 2026 |

---

## Learning Objectives

By completing this tutorial, you will be able to:

1. **Understand** AutoML for FL
2. **Implement** federated hyperparameter optimization
3. **Design** automated FL pipelines
4. **Analyze** search strategies
5. **Deploy** auto-configured FL systems

---

## Prerequisites

Before starting this tutorial, ensure you have:

- Completed tutorials 001-122
- Understanding of FL fundamentals
- Knowledge of hyperparameter optimization
- Familiarity with AutoML concepts

---

## Background and Theory

### Why AutoML for FL?

FL has many hyperparameters:
- Learning rate, batch size, epochs
- Client selection parameters
- Aggregation strategies
- Compression settings

Manual tuning is difficult due to:
- Distributed data (can't validate centrally)
- High experiment costs
- Client heterogeneity

### AutoML Components

```
Federated AutoML:
├── Hyperparameter Optimization
│   ├── Grid search (expensive)
│   ├── Random search
│   ├── Bayesian optimization
│   └── Population-based training
├── Architecture Search
│   ├── Neural architecture search
│   ├── Layer selection
│   └── Width/depth optimization
├── Algorithm Selection
│   ├── Aggregation strategy
│   ├── Optimizer selection
│   └── Client selection
└── Pipeline Automation
    ├── Data preprocessing
    ├── Feature engineering
    └── Model deployment
```

---

## Implementation Code

```python
#!/usr/bin/env python3
"""
Tutorial 123: Federated Learning AutoML

This module implements automated hyperparameter optimization
and model selection for federated learning.

Copyright 2026 Olaf Yunus Laitinen Imanov and Contributors
Released under EUPL 1.2
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Callable
import copy
import logging
from enum import Enum

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class AutoMLConfig:
    """Configuration for AutoML FL."""

    # Search parameters
    num_trials: int = 20
    num_rounds_per_trial: int = 15

    # FL parameters
    num_clients: int = 15
    clients_per_round: int = 8

    # Model parameters
    input_dim: int = 32
    num_classes: int = 10

    seed: int = 42


@dataclass
class SearchSpace:
    """Hyperparameter search space."""

    learning_rate: Tuple[float, float] = (0.001, 0.1)
    hidden_dim: Tuple[int, int] = (32, 128)
    num_layers: Tuple[int, int] = (1, 3)
    batch_size: List[int] = field(default_factory=lambda: [16, 32, 64])
    local_epochs: Tuple[int, int] = (1, 5)
    dropout: Tuple[float, float] = (0.0, 0.5)


class HyperparameterSampler:
    """Sample hyperparameters from search space."""

    def __init__(self, space: SearchSpace, seed: int = 0):
        self.space = space
        self.rng = np.random.RandomState(seed)

    def sample(self) -> Dict[str, Any]:
        """Sample random configuration."""
        return {
            "learning_rate": self.rng.uniform(*self.space.learning_rate),
            "hidden_dim": self.rng.randint(*self.space.hidden_dim),
            "num_layers": self.rng.randint(*self.space.num_layers),
            "batch_size": self.rng.choice(self.space.batch_size),
            "local_epochs": self.rng.randint(*self.space.local_epochs),
            "dropout": self.rng.uniform(*self.space.dropout)
        }

    def sample_grid(self, n: int) -> List[Dict[str, Any]]:
        """Sample n configurations."""
        return [self.sample() for _ in range(n)]


class BayesianOptimizer:
    """Simple Bayesian optimization for HPO."""

    def __init__(self, space: SearchSpace, seed: int = 0):
        self.space = space
        self.rng = np.random.RandomState(seed)
        self.observations: List[Tuple[Dict, float]] = []

    def suggest(self) -> Dict[str, Any]:
        """Suggest next configuration."""
        if len(self.observations) < 5:
            # Random exploration
            return HyperparameterSampler(self.space, self.rng.randint(10000)).sample()

        # Find best and explore nearby
        best_config, best_score = max(self.observations, key=lambda x: x[1])

        # Perturb best config
        new_config = {}
        for key, value in best_config.items():
            if key == "learning_rate":
                new_config[key] = np.clip(
                    value * self.rng.uniform(0.8, 1.2),
                    *self.space.learning_rate
                )
            elif key == "hidden_dim":
                new_config[key] = int(np.clip(
                    value + self.rng.randint(-16, 17),
                    *self.space.hidden_dim
                ))
            elif key == "num_layers":
                new_config[key] = int(np.clip(
                    value + self.rng.choice([-1, 0, 1]),
                    *self.space.num_layers
                ))
            elif key == "batch_size":
                new_config[key] = self.rng.choice(self.space.batch_size)
            elif key == "local_epochs":
                new_config[key] = int(np.clip(
                    value + self.rng.choice([-1, 0, 1]),
                    *self.space.local_epochs
                ))
            elif key == "dropout":
                new_config[key] = np.clip(
                    value + self.rng.uniform(-0.1, 0.1),
                    *self.space.dropout
                )
            else:
                new_config[key] = value

        return new_config

    def observe(self, config: Dict, score: float) -> None:
        """Record observation."""
        self.observations.append((config, score))


class AutoMLDataset(Dataset):
    """Dataset for AutoML experiments."""

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


def create_model(config: Dict, automl_config: AutoMLConfig) -> nn.Module:
    """Create model from hyperparameters."""
    layers = []

    in_dim = automl_config.input_dim

    for _ in range(config["num_layers"]):
        layers.append(nn.Linear(in_dim, config["hidden_dim"]))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(config["dropout"]))
        in_dim = config["hidden_dim"]

    layers.append(nn.Linear(in_dim, automl_config.num_classes))

    return nn.Sequential(*layers)


class AutoMLClient:
    """Client for AutoML experiments."""

    def __init__(
        self,
        client_id: int,
        dataset: AutoMLDataset,
        automl_config: AutoMLConfig
    ):
        self.client_id = client_id
        self.dataset = dataset
        self.automl_config = automl_config

    def train(
        self,
        model: nn.Module,
        hyperparams: Dict
    ) -> Dict[str, Any]:
        """Train with given hyperparameters."""
        local = copy.deepcopy(model)
        optimizer = torch.optim.Adam(
            local.parameters(),
            lr=hyperparams["learning_rate"]
        )

        loader = DataLoader(
            self.dataset,
            batch_size=hyperparams["batch_size"],
            shuffle=True
        )

        local.train()
        total_loss = 0.0
        num_batches = 0

        for _ in range(hyperparams["local_epochs"]):
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


class AutoMLServer:
    """Server with AutoML capabilities."""

    def __init__(
        self,
        clients: List[AutoMLClient],
        test_data: AutoMLDataset,
        automl_config: AutoMLConfig
    ):
        self.clients = clients
        self.test_data = test_data
        self.automl_config = automl_config

        self.optimizer = BayesianOptimizer(SearchSpace(), automl_config.seed)
        self.best_config: Optional[Dict] = None
        self.best_score: float = 0.0
        self.trial_history: List[Dict] = []

    def evaluate(self, model: nn.Module) -> float:
        """Evaluate model."""
        model.eval()
        loader = DataLoader(self.test_data, batch_size=64)

        correct, total = 0, 0
        with torch.no_grad():
            for x, y in loader:
                pred = model(x).argmax(dim=1)
                correct += (pred == y).sum().item()
                total += len(y)

        return correct / total

    def run_trial(self, hyperparams: Dict) -> float:
        """Run single FL trial with given hyperparams."""
        model = create_model(hyperparams, self.automl_config)

        for round_num in range(self.automl_config.num_rounds_per_trial):
            # Select clients
            n = min(self.automl_config.clients_per_round, len(self.clients))
            indices = np.random.choice(len(self.clients), n, replace=False)
            selected = [self.clients[i] for i in indices]

            # Collect updates
            updates = [c.train(model, hyperparams) for c in selected]

            # Aggregate
            total_samples = sum(u["num_samples"] for u in updates)
            new_state = {}

            for key in updates[0]["state_dict"]:
                new_state[key] = sum(
                    (u["num_samples"] / total_samples) * u["state_dict"][key].float()
                    for u in updates
                )

            model.load_state_dict(new_state)

        return self.evaluate(model)

    def search(self) -> Dict:
        """Run AutoML search."""
        logger.info(f"Starting AutoML with {self.automl_config.num_trials} trials")

        for trial_num in range(self.automl_config.num_trials):
            # Get config
            hyperparams = self.optimizer.suggest()

            # Run trial
            score = self.run_trial(hyperparams)

            # Record
            self.optimizer.observe(hyperparams, score)

            self.trial_history.append({
                "trial": trial_num,
                "hyperparams": hyperparams,
                "score": score
            })

            # Update best
            if score > self.best_score:
                self.best_score = score
                self.best_config = hyperparams
                logger.info(f"New best! Trial {trial_num}: {score:.4f}")

            if (trial_num + 1) % 5 == 0:
                logger.info(
                    f"Trial {trial_num + 1}/{self.automl_config.num_trials}: "
                    f"score={score:.4f}, best={self.best_score:.4f}"
                )

        return self.best_config


def main():
    """Main entry point."""
    print("=" * 60)
    print("Tutorial 123: FL AutoML")
    print("=" * 60)

    automl_config = AutoMLConfig()
    torch.manual_seed(automl_config.seed)
    np.random.seed(automl_config.seed)

    # Create clients
    clients = []
    for i in range(automl_config.num_clients):
        dataset = AutoMLDataset(
            client_id=i,
            dim=automl_config.input_dim,
            seed=automl_config.seed
        )
        client = AutoMLClient(i, dataset, automl_config)
        clients.append(client)

    test_data = AutoMLDataset(client_id=999, n=300, seed=999)

    # Run AutoML
    server = AutoMLServer(clients, test_data, automl_config)
    best_config = server.search()

    print("\n" + "=" * 60)
    print("AutoML Complete")
    print(f"Best Score: {server.best_score:.4f}")
    print("\nBest Configuration:")
    for key, value in best_config.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    print("=" * 60)


if __name__ == "__main__":
    main()
```

---

## Key Insights

### AutoML Challenges in FL

1. **Distributed validation**: No central test set
2. **Expensive trials**: Each trial is full FL
3. **Client heterogeneity**: Optimal differs per client
4. **Communication cost**: Search adds overhead

### Best Practices

- Use efficient search (Bayesian, PBT)
- Early stopping for bad configurations
- Consider client-specific tuning
- Cache trained models

---

## Exercises

1. **Exercise 1**: Implement population-based training
2. **Exercise 2**: Add architecture search
3. **Exercise 3**: Design client-aware HPO
4. **Exercise 4**: Implement early stopping

---

## References

1. Khodak, M., et al. (2021). Federated hyperparameter tuning. In *ICLR*.
2. Shu, R., et al. (2022). AutoFL: Towards automated FL. *IEEE TNNLS*.
3. Dai, Z., et al. (2022). Fed-HPO: Efficient HPO for FL. In *KDD*.

---

*Copyright 2026 Olaf Yunus Laitinen Imanov and Contributors. Released under EUPL 1.2.*
