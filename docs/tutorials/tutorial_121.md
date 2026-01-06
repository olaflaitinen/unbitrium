# Tutorial 121: FL Hyperparameter Optimization

---

## Metadata

| Property | Value |
|----------|-------|
| **Tutorial ID** | 121 |
| **Title** | Federated Learning Hyperparameter Optimization |
| **Category** | Optimization |
| **Difficulty** | Advanced |
| **Duration** | 120 minutes |
| **Prerequisites** | Tutorial 001-120 |
| **Author** | Unbitrium Contributors |
| **Last Updated** | January 2026 |

---

## Learning Objectives

By completing this tutorial, you will be able to:

1. **Understand** HPO challenges in FL
2. **Implement** federated hyperparameter tuning
3. **Design** distributed search strategies
4. **Analyze** search space efficiency
5. **Deploy** auto-tuned FL systems

---

## Prerequisites

Before starting this tutorial, ensure you have:

- Completed tutorials 001-120
- Understanding of FL fundamentals
- Knowledge of hyperparameter optimization
- Familiarity with search algorithms

---

## Background and Theory

### FL HPO Challenges

HPO in FL is harder than centralized:
- No central validation set
- Each trial requires full FL training
- Client heterogeneity affects optima
- High computational cost

### Search Strategies

```
FL HPO Strategies:
├── Grid Search
│   ├── Exhaustive
│   ├── Expensive
│   └── Good for small spaces
├── Random Search
│   ├── More efficient
│   ├── Covers more ground
│   └── Better for large spaces
├── Bayesian Optimization
│   ├── Surrogate model
│   ├── Acquisition function
│   └── Sample efficient
└── Population-Based
    ├── Evolutionary
    ├── Parallel trials
    └── Adaptive
```

---

## Implementation Code

```python
#!/usr/bin/env python3
"""
Tutorial 121: FL Hyperparameter Optimization

This module implements hyperparameter optimization
for federated learning systems.

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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class HPOConfig:
    """Configuration for FL HPO."""

    num_trials: int = 15
    num_rounds_per_trial: int = 10

    num_clients: int = 10
    clients_per_round: int = 5

    input_dim: int = 32
    num_classes: int = 10

    seed: int = 42


@dataclass
class HyperparameterSpace:
    """Define hyperparameter search space."""

    learning_rate: Tuple[float, float] = (0.001, 0.1)
    hidden_dim: List[int] = field(default_factory=lambda: [32, 64, 128])
    batch_size: List[int] = field(default_factory=lambda: [16, 32, 64])
    local_epochs: List[int] = field(default_factory=lambda: [1, 2, 3, 5])
    dropout: Tuple[float, float] = (0.0, 0.5)
    weight_decay: Tuple[float, float] = (0.0, 0.01)


class RandomSearcher:
    """Random hyperparameter search."""

    def __init__(self, space: HyperparameterSpace, seed: int = 0):
        self.space = space
        self.rng = np.random.RandomState(seed)

    def sample(self) -> Dict[str, Any]:
        """Sample random configuration."""
        return {
            "learning_rate": np.exp(self.rng.uniform(
                np.log(self.space.learning_rate[0]),
                np.log(self.space.learning_rate[1])
            )),
            "hidden_dim": self.rng.choice(self.space.hidden_dim),
            "batch_size": self.rng.choice(self.space.batch_size),
            "local_epochs": self.rng.choice(self.space.local_epochs),
            "dropout": self.rng.uniform(*self.space.dropout),
            "weight_decay": self.rng.uniform(*self.space.weight_decay)
        }


class GaussianProcessSurrogate:
    """Simple GP surrogate for Bayesian optimization."""

    def __init__(self, input_dim: int = 6):
        self.X: List[np.ndarray] = []
        self.y: List[float] = []
        self.input_dim = input_dim

    def add_observation(self, x: np.ndarray, y: float) -> None:
        """Add observation."""
        self.X.append(x)
        self.y.append(y)

    def predict(self, x: np.ndarray) -> Tuple[float, float]:
        """Predict mean and std (simplified)."""
        if len(self.X) < 3:
            return 0.5, 1.0

        # Simple nearest neighbor estimate
        X = np.array(self.X)
        y = np.array(self.y)

        distances = np.linalg.norm(X - x, axis=1)
        k = min(3, len(self.X))
        nearest_idx = np.argsort(distances)[:k]

        mean = y[nearest_idx].mean()
        std = max(0.1, y[nearest_idx].std())

        return mean, std

    def expected_improvement(
        self,
        x: np.ndarray,
        best_y: float
    ) -> float:
        """Compute expected improvement."""
        mean, std = self.predict(x)

        if std == 0:
            return 0.0

        z = (mean - best_y) / std
        ei = (mean - best_y) * self._cdf(z) + std * self._pdf(z)

        return max(0, ei)

    def _pdf(self, x: float) -> float:
        """Standard normal PDF."""
        return np.exp(-0.5 * x ** 2) / np.sqrt(2 * np.pi)

    def _cdf(self, x: float) -> float:
        """Standard normal CDF (approximation)."""
        return 0.5 * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))


class BayesianSearcher:
    """Bayesian hyperparameter search."""

    def __init__(self, space: HyperparameterSpace, seed: int = 0):
        self.space = space
        self.rng = np.random.RandomState(seed)
        self.surrogate = GaussianProcessSurrogate()
        self.observations: List[Tuple[Dict, float]] = []

    def _config_to_vector(self, config: Dict) -> np.ndarray:
        """Convert config to vector."""
        return np.array([
            np.log(config["learning_rate"]),
            config["hidden_dim"] / 128,
            config["batch_size"] / 64,
            config["local_epochs"] / 5,
            config["dropout"],
            config["weight_decay"] * 100
        ])

    def sample(self) -> Dict[str, Any]:
        """Sample next configuration."""
        if len(self.observations) < 5:
            # Random exploration
            return RandomSearcher(self.space, self.rng.randint(10000)).sample()

        # Generate candidates
        candidates = []
        for _ in range(100):
            config = RandomSearcher(self.space, self.rng.randint(10000)).sample()
            x = self._config_to_vector(config)

            best_y = max(y for _, y in self.observations)
            ei = self.surrogate.expected_improvement(x, best_y)

            candidates.append((config, ei))

        # Select best
        candidates.sort(key=lambda c: c[1], reverse=True)
        return candidates[0][0]

    def observe(self, config: Dict, score: float) -> None:
        """Record observation."""
        self.observations.append((config, score))
        x = self._config_to_vector(config)
        self.surrogate.add_observation(x, score)


class HPODataset(Dataset):
    """Dataset for HPO experiments."""

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


def create_model(config: Dict, input_dim: int, num_classes: int) -> nn.Module:
    """Create model from hyperparameters."""
    return nn.Sequential(
        nn.Linear(input_dim, config["hidden_dim"]),
        nn.ReLU(),
        nn.Dropout(config["dropout"]),
        nn.Linear(config["hidden_dim"], num_classes)
    )


class HPOClient:
    """Client for HPO experiments."""

    def __init__(
        self,
        client_id: int,
        dataset: HPODataset,
        hpo_config: HPOConfig
    ):
        self.client_id = client_id
        self.dataset = dataset
        self.hpo_config = hpo_config

    def train(
        self,
        model: nn.Module,
        hyperparams: Dict
    ) -> Dict[str, Any]:
        """Train with given hyperparameters."""
        local = copy.deepcopy(model)
        optimizer = torch.optim.Adam(
            local.parameters(),
            lr=hyperparams["learning_rate"],
            weight_decay=hyperparams["weight_decay"]
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
            "avg_loss": total_loss / num_batches
        }


class HPOServer:
    """Server for FL HPO."""

    def __init__(
        self,
        clients: List[HPOClient],
        test_data: HPODataset,
        hpo_config: HPOConfig
    ):
        self.clients = clients
        self.test_data = test_data
        self.hpo_config = hpo_config

        self.searcher = BayesianSearcher(HyperparameterSpace(), hpo_config.seed)
        self.trial_history: List[Dict] = []
        self.best_config: Optional[Dict] = None
        self.best_score: float = 0.0

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
        """Run single FL trial."""
        model = create_model(
            hyperparams,
            self.hpo_config.input_dim,
            self.hpo_config.num_classes
        )

        for _ in range(self.hpo_config.num_rounds_per_trial):
            n = min(self.hpo_config.clients_per_round, len(self.clients))
            indices = np.random.choice(len(self.clients), n, replace=False)
            selected = [self.clients[i] for i in indices]

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

    def search(self) -> Dict[str, Any]:
        """Run HPO search."""
        logger.info(f"Starting FL HPO with {self.hpo_config.num_trials} trials")

        for trial_num in range(self.hpo_config.num_trials):
            config = self.searcher.sample()
            score = self.run_trial(config)

            self.searcher.observe(config, score)

            self.trial_history.append({
                "trial": trial_num,
                "config": config,
                "score": score
            })

            if score > self.best_score:
                self.best_score = score
                self.best_config = config
                logger.info(f"New best at trial {trial_num}: {score:.4f}")

        return self.best_config


def main():
    """Main entry point."""
    print("=" * 60)
    print("Tutorial 121: FL HPO")
    print("=" * 60)

    hpo_config = HPOConfig()
    torch.manual_seed(hpo_config.seed)
    np.random.seed(hpo_config.seed)

    # Create clients
    clients = []
    for i in range(hpo_config.num_clients):
        dataset = HPODataset(client_id=i, dim=hpo_config.input_dim, seed=hpo_config.seed)
        client = HPOClient(i, dataset, hpo_config)
        clients.append(client)

    test_data = HPODataset(client_id=999, n=300, seed=999)

    # Run HPO
    server = HPOServer(clients, test_data, hpo_config)
    best_config = server.search()

    print("\n" + "=" * 60)
    print("HPO Complete")
    print(f"Best Score: {server.best_score:.4f}")
    print("\nBest Configuration:")
    for key, value in best_config.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.6f}")
        else:
            print(f"  {key}: {value}")
    print("=" * 60)


if __name__ == "__main__":
    main()
```

---

## Key Insights

### HPO Best Practices

1. **Start with random**: Good baseline
2. **Use Bayesian**: Sample efficient
3. **Early stopping**: Save compute
4. **Multi-fidelity**: Quick trials first

---

## Exercises

1. **Exercise 1**: Add multi-fidelity optimization
2. **Exercise 2**: Implement BOHB
3. **Exercise 3**: Design client-aware HPO
4. **Exercise 4**: Add early stopping

---

## References

1. Khodak, M., et al. (2021). Federated hyperparameter tuning. In *ICLR*.
2. Falkner, S., et al. (2018). BOHB: Robust hyperparameter optimization. In *ICML*.
3. Li, L., et al. (2017). Hyperband: Principled hyperparameter optimization. In *JMLR*.

---

*Copyright 2026 Olaf Yunus Laitinen Imanov and Contributors. Released under EUPL 1.2.*
