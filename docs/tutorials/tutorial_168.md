# Tutorial 168: FL Advanced Optimization

---

## Metadata

| Property | Value |
|----------|-------|
| **Tutorial ID** | 168 |
| **Title** | Federated Learning Advanced Optimization |
| **Category** | Optimization |
| **Difficulty** | Expert |
| **Duration** | 150 minutes |
| **Prerequisites** | Tutorial 001-167 |
| **Author** | Unbitrium Contributors |
| **Last Updated** | January 2026 |

---

## Learning Objectives

By completing this tutorial, you will be able to:

1. **Master** advanced FL optimization techniques
2. **Implement** SCAFFOLD, FedProx, and FedNova
3. **Design** adaptive optimization strategies
4. **Analyze** convergence in heterogeneous settings
5. **Deploy** state-of-the-art FL optimizers

---

## Prerequisites

Before starting this tutorial, ensure you have:

- Completed tutorials 001-167
- Strong understanding of optimization
- Knowledge of FL challenges
- Familiarity with adaptive methods

---

## Background and Theory

### Optimization Challenges in FL

FL faces unique optimization challenges:
- Non-IID data distributions
- Client drift during local training
- Partial client participation
- Communication constraints

### Advanced Optimizers

```
FL Optimizer Landscape:
├── Variance Reduction
│   ├── SCAFFOLD
│   └── VRL-SGD
├── Proximal Methods
│   ├── FedProx
│   └── pFedMe
├── Normalized Averaging
│   ├── FedNova
│   └── FedOpt
└── Adaptive Methods
    ├── FedAdam
    ├── FedYogi
    └── FedAdagrad
```

### Comparison

| Method | Client Drift | Non-IID | Communication |
|--------|--------------|---------|---------------|
| FedAvg | ❌ | ❌ | ✅ |
| FedProx | ✅ | ❌ | ✅ |
| SCAFFOLD | ✅ | ✅ | ⚠️ |
| FedNova | ❌ | ✅ | ✅ |

---

## Implementation Code

```python
#!/usr/bin/env python3
"""
Tutorial 168: Federated Learning Advanced Optimization

This module implements state-of-the-art FL optimizers including
SCAFFOLD, FedProx, FedNova, and adaptive server optimizers.

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
from abc import ABC, abstractmethod
import copy
import logging
from enum import Enum

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OptimizerType(Enum):
    """Available FL optimizers."""
    FEDAVG = "fedavg"
    FEDPROX = "fedprox"
    SCAFFOLD = "scaffold"
    FEDNOVA = "fednova"
    FEDADAM = "fedadam"


@dataclass
class AdvOptConfig:
    """Configuration for advanced optimization."""

    num_rounds: int = 50
    num_clients: int = 20
    clients_per_round: int = 10

    input_dim: int = 32
    hidden_dim: int = 64
    num_classes: int = 10

    learning_rate: float = 0.01
    batch_size: int = 32
    local_epochs: int = 5

    # Optimizer-specific
    optimizer_type: OptimizerType = OptimizerType.SCAFFOLD
    fedprox_mu: float = 0.01
    server_lr: float = 1.0

    # Non-IID
    dirichlet_alpha: float = 0.5

    seed: int = 42


class OptDataset(Dataset):
    """Dataset with configurable heterogeneity."""

    def __init__(
        self,
        client_id: int,
        n: int = 200,
        dim: int = 32,
        classes: int = 10,
        seed: int = 0,
        class_probs: Optional[np.ndarray] = None
    ):
        np.random.seed(seed + client_id)

        if class_probs is None:
            class_probs = np.ones(classes) / classes

        self.x = torch.randn(n, dim, dtype=torch.float32)
        self.y = torch.tensor(
            np.random.choice(classes, n, p=class_probs),
            dtype=torch.long
        )

        for i in range(n):
            self.x[i, self.y[i].item() % dim] += 2.0

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx]


class OptModel(nn.Module):
    """Standard model for optimization testing."""

    def __init__(self, config: AdvOptConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# =============================================================================
# Client Implementations
# =============================================================================

class BaseClient(ABC):
    """Base client class."""

    def __init__(
        self,
        client_id: int,
        dataset: OptDataset,
        config: AdvOptConfig
    ):
        self.client_id = client_id
        self.dataset = dataset
        self.config = config

    @abstractmethod
    def train(self, model: nn.Module, **kwargs) -> Dict[str, Any]:
        pass


class FedAvgClient(BaseClient):
    """Standard FedAvg client."""

    def train(self, model: nn.Module, **kwargs) -> Dict[str, Any]:
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


class FedProxClient(BaseClient):
    """FedProx client with proximal term."""

    def train(self, model: nn.Module, **kwargs) -> Dict[str, Any]:
        local = copy.deepcopy(model)
        global_params = {
            k: v.clone() for k, v in model.state_dict().items()
        }

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

        mu = self.config.fedprox_mu

        for _ in range(self.config.local_epochs):
            for x, y in loader:
                optimizer.zero_grad()

                # Standard loss
                loss = F.cross_entropy(local(x), y)

                # Proximal term
                prox_term = 0.0
                for name, param in local.named_parameters():
                    prox_term += ((param - global_params[name].detach()) ** 2).sum()

                loss = loss + (mu / 2) * prox_term

                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                num_batches += 1

        return {
            "state_dict": {k: v.cpu() for k, v in local.state_dict().items()},
            "num_samples": len(self.dataset),
            "avg_loss": total_loss / num_batches
        }


class SCAFFOLDClient(BaseClient):
    """SCAFFOLD client with control variates."""

    def __init__(
        self,
        client_id: int,
        dataset: OptDataset,
        config: AdvOptConfig
    ):
        super().__init__(client_id, dataset, config)
        self.c_local: Optional[Dict[str, torch.Tensor]] = None

    def train(
        self,
        model: nn.Module,
        c_global: Optional[Dict[str, torch.Tensor]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        local = copy.deepcopy(model)
        global_state = {k: v.clone() for k, v in model.state_dict().items()}

        # Initialize control variates if needed
        if self.c_local is None:
            self.c_local = {
                k: torch.zeros_like(v) for k, v in model.state_dict().items()
            }
        if c_global is None:
            c_global = {
                k: torch.zeros_like(v) for k, v in model.state_dict().items()
            }

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
        total_steps = 0

        for _ in range(self.config.local_epochs):
            for x, y in loader:
                optimizer.zero_grad()
                loss = F.cross_entropy(local(x), y)
                loss.backward()

                # Apply control variate correction
                with torch.no_grad():
                    for name, param in local.named_parameters():
                        if param.grad is not None:
                            correction = c_global[name] - self.c_local[name]
                            param.grad.add_(correction)

                optimizer.step()

                total_loss += loss.item()
                num_batches += 1
                total_steps += 1

        # Update local control variate
        c_local_new = {}
        with torch.no_grad():
            for name, param in local.named_parameters():
                c_local_new[name] = self.c_local[name] - c_global[name] + (
                    global_state[name] - param.data
                ) / (total_steps * self.config.learning_rate)

        # Compute delta c
        delta_c = {
            name: c_local_new[name] - self.c_local[name]
            for name in c_local_new
        }

        self.c_local = c_local_new

        return {
            "state_dict": {k: v.cpu() for k, v in local.state_dict().items()},
            "delta_c": delta_c,
            "num_samples": len(self.dataset),
            "avg_loss": total_loss / num_batches
        }


class FedNovaClient(BaseClient):
    """FedNova client with normalized averaging."""

    def train(self, model: nn.Module, **kwargs) -> Dict[str, Any]:
        local = copy.deepcopy(model)
        initial_state = {k: v.clone() for k, v in model.state_dict().items()}

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
        total_steps = 0

        for _ in range(self.config.local_epochs):
            for x, y in loader:
                optimizer.zero_grad()
                loss = F.cross_entropy(local(x), y)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                num_batches += 1
                total_steps += 1

        # Compute normalized update
        with torch.no_grad():
            delta = {}
            for name, param in local.named_parameters():
                delta[name] = (initial_state[name] - param.data) / total_steps

        return {
            "delta": delta,
            "tau": total_steps,  # Normalization factor
            "num_samples": len(self.dataset),
            "avg_loss": total_loss / num_batches
        }


# =============================================================================
# Server Implementations
# =============================================================================

class FedServer:
    """Server with multiple aggregation strategies."""

    def __init__(
        self,
        model: nn.Module,
        clients: List[BaseClient],
        test_data: OptDataset,
        config: AdvOptConfig
    ):
        self.model = model
        self.clients = clients
        self.test_data = test_data
        self.config = config

        # SCAFFOLD control variate
        self.c_global: Dict[str, torch.Tensor] = {
            k: torch.zeros_like(v) for k, v in model.state_dict().items()
        }

        # Server optimizer state (for FedAdam)
        self.m = {k: torch.zeros_like(v) for k, v in model.state_dict().items()}
        self.v = {k: torch.zeros_like(v) for k, v in model.state_dict().items()}
        self.t = 0

        self.history: List[Dict] = []

    def aggregate_fedavg(self, updates: List[Dict]) -> None:
        """Standard FedAvg aggregation."""
        total_samples = sum(u["num_samples"] for u in updates)
        new_state = {}

        for key in updates[0]["state_dict"]:
            new_state[key] = sum(
                (u["num_samples"] / total_samples) * u["state_dict"][key].float()
                for u in updates
            )

        self.model.load_state_dict(new_state)

    def aggregate_scaffold(self, updates: List[Dict]) -> None:
        """SCAFFOLD aggregation with control variate updates."""
        total_samples = sum(u["num_samples"] for u in updates)

        # Aggregate model updates
        new_state = {}
        for key in updates[0]["state_dict"]:
            new_state[key] = sum(
                (u["num_samples"] / total_samples) * u["state_dict"][key].float()
                for u in updates
            )
        self.model.load_state_dict(new_state)

        # Update global control variate
        n_clients = len(self.clients)
        for key in self.c_global:
            delta_c_sum = sum(
                u["delta_c"][key].float() for u in updates
            )
            self.c_global[key] += (len(updates) / n_clients) * delta_c_sum

    def aggregate_fednova(self, updates: List[Dict]) -> None:
        """FedNova normalized aggregation."""
        total_tau = sum(u["tau"] for u in updates)

        with torch.no_grad():
            for name, param in self.model.named_parameters():
                delta = sum(
                    u["tau"] / total_tau * u["delta"][name].float()
                    for u in updates
                )
                param.data -= self.config.server_lr * delta

    def aggregate_fedadam(self, updates: List[Dict]) -> None:
        """FedAdam with server-side Adam."""
        total_samples = sum(u["num_samples"] for u in updates)
        self.t += 1

        # Compute pseudo-gradient
        current = self.model.state_dict()
        avg_update = {}

        for key in updates[0]["state_dict"]:
            avg = sum(
                (u["num_samples"] / total_samples) * u["state_dict"][key].float()
                for u in updates
            )
            avg_update[key] = current[key] - avg

        # Apply Adam update
        beta1, beta2 = 0.9, 0.99
        eps = 1e-8

        new_state = {}
        for key in current:
            self.m[key] = beta1 * self.m[key] + (1 - beta1) * avg_update[key]
            self.v[key] = beta2 * self.v[key] + (1 - beta2) * (avg_update[key] ** 2)

            m_hat = self.m[key] / (1 - beta1 ** self.t)
            v_hat = self.v[key] / (1 - beta2 ** self.t)

            new_state[key] = current[key] - self.config.server_lr * m_hat / (torch.sqrt(v_hat) + eps)

        self.model.load_state_dict(new_state)

    def evaluate(self) -> Dict[str, float]:
        """Evaluate model."""
        self.model.eval()
        loader = DataLoader(self.test_data, batch_size=64)

        correct, total = 0, 0
        total_loss = 0.0

        with torch.no_grad():
            for x, y in loader:
                output = self.model(x)
                loss = F.cross_entropy(output, y)
                pred = output.argmax(dim=1)

                correct += (pred == y).sum().item()
                total += len(y)
                total_loss += loss.item() * len(y)

        return {"accuracy": correct / total, "loss": total_loss / total}

    def train(self) -> List[Dict]:
        """Run training with selected optimizer."""
        opt_type = self.config.optimizer_type
        logger.info(f"Starting FL with {opt_type.value}")

        for round_num in range(self.config.num_rounds):
            # Select clients
            n = min(self.config.clients_per_round, len(self.clients))
            indices = np.random.choice(len(self.clients), n, replace=False)
            selected = [self.clients[i] for i in indices]

            # Collect updates
            updates = []
            for client in selected:
                if opt_type == OptimizerType.SCAFFOLD:
                    update = client.train(self.model, c_global=self.c_global)
                else:
                    update = client.train(self.model)
                updates.append(update)

            # Aggregate
            if opt_type == OptimizerType.FEDAVG or opt_type == OptimizerType.FEDPROX:
                self.aggregate_fedavg(updates)
            elif opt_type == OptimizerType.SCAFFOLD:
                self.aggregate_scaffold(updates)
            elif opt_type == OptimizerType.FEDNOVA:
                self.aggregate_fednova(updates)
            elif opt_type == OptimizerType.FEDADAM:
                self.aggregate_fedadam(updates)

            # Evaluate
            metrics = self.evaluate()

            record = {"round": round_num, **metrics}
            self.history.append(record)

            if (round_num + 1) % 10 == 0:
                logger.info(f"Round {round_num + 1}: acc={metrics['accuracy']:.4f}")

        return self.history


def create_non_iid_datasets(
    num_clients: int,
    config: AdvOptConfig
) -> List[OptDataset]:
    """Create non-IID datasets using Dirichlet distribution."""
    datasets = []

    for i in range(num_clients):
        class_probs = np.random.dirichlet(
            [config.dirichlet_alpha] * config.num_classes
        )
        dataset = OptDataset(
            client_id=i,
            dim=config.input_dim,
            classes=config.num_classes,
            seed=config.seed,
            class_probs=class_probs
        )
        datasets.append(dataset)

    return datasets


def main():
    """Main entry point."""
    print("=" * 60)
    print("Tutorial 168: FL Advanced Optimization")
    print("=" * 60)

    config = AdvOptConfig()
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    # Create non-IID datasets
    datasets = create_non_iid_datasets(config.num_clients, config)

    # Create appropriate clients
    ClientClass = {
        OptimizerType.FEDAVG: FedAvgClient,
        OptimizerType.FEDPROX: FedProxClient,
        OptimizerType.SCAFFOLD: SCAFFOLDClient,
        OptimizerType.FEDNOVA: FedNovaClient,
        OptimizerType.FEDADAM: FedAvgClient
    }[config.optimizer_type]

    clients = [ClientClass(i, d, config) for i, d in enumerate(datasets)]

    test_data = OptDataset(client_id=999, n=500, seed=999)
    model = OptModel(config)

    server = FedServer(model, clients, test_data, config)
    history = server.train()

    print("\n" + "=" * 60)
    print(f"Optimizer: {config.optimizer_type.value}")
    print(f"Final Accuracy: {history[-1]['accuracy']:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
```

---

## Key Insights

### Optimizer Selection Guide

1. **FedAvg**: Use for IID data, simple setup
2. **FedProx**: Use when client drift is a concern
3. **SCAFFOLD**: Use for highly non-IID data
4. **FedNova**: Use with heterogeneous local epochs
5. **FedAdam**: Use for faster convergence

---

## Exercises

1. **Exercise 1**: Compare all optimizers on same dataset
2. **Exercise 2**: Tune hyperparameters for each optimizer
3. **Exercise 3**: Implement FedYogi optimizer
4. **Exercise 4**: Add momentum to SCAFFOLD

---

## References

1. Karimireddy, S.P., et al. (2020). SCAFFOLD: Stochastic controlled averaging. In *ICML*.
2. Li, T., et al. (2020). FedProx: Federated optimization in heterogeneous networks. In *MLSys*.
3. Wang, J., et al. (2020). Tackling the objective inconsistency problem in heterogeneous FL. In *NeurIPS*.

---

*Copyright 2026 Olaf Yunus Laitinen Imanov and Contributors. Released under EUPL 1.2.*
