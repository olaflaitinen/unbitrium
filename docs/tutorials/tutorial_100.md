# Tutorial 100: FL Multi-Party Computation

---

## Metadata

| Property | Value |
|----------|-------|
| **Tutorial ID** | 100 |
| **Title** | FL Multi-Party Computation |
| **Category** | Privacy |
| **Difficulty** | Expert |
| **Duration** | 120 minutes |
| **Prerequisites** | Tutorial 001-099 |
| **Author** | Unbitrium Contributors |
| **Last Updated** | January 2026 |

---

## Learning Objectives

By completing this tutorial, you will be able to:

1. **Understand** MPC fundamentals
2. **Implement** MPC-based aggregation
3. **Design** secure computation protocols
4. **Analyze** MPC security
5. **Deploy** MPC in FL

---

## Background and Theory

### MPC Concepts

```
MPC Components:
├── Secret Sharing
│   ├── Additive sharing
│   ├── Shamir sharing
│   └── Replicated sharing
├── Protocols
│   ├── Garbled circuits
│   ├── GMW protocol
│   └── BGW protocol
├── Operations
│   ├── Secure addition
│   ├── Secure multiplication
│   └── Secure comparison
└── FL Integration
    ├── Distributed aggregation
    ├── Threshold decryption
    └── Verifiable computation
```

---

## Implementation Code

```python
#!/usr/bin/env python3
"""
Tutorial 100: FL Multi-Party Computation

This module implements MPC-based secure aggregation
for federated learning.

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
class MPCConfig:
    """MPC configuration."""

    num_rounds: int = 30
    num_clients: int = 10
    clients_per_round: int = 5

    num_servers: int = 3  # MPC servers
    threshold: int = 2    # Reconstruction threshold

    input_dim: int = 32
    hidden_dim: int = 64
    num_classes: int = 10

    learning_rate: float = 0.01
    batch_size: int = 32
    local_epochs: int = 3

    seed: int = 42


class SecretSharing:
    """Additive secret sharing."""

    def __init__(self, num_parties: int, threshold: int):
        self.num_parties = num_parties
        self.threshold = threshold
        self.prime = 2 ** 31 - 1

    def share(self, secret: torch.Tensor) -> List[torch.Tensor]:
        """Create additive shares."""
        shares = []
        running_sum = torch.zeros_like(secret)

        for i in range(self.num_parties - 1):
            share = torch.randint_like(secret, 0, 10000).float()
            shares.append(share)
            running_sum += share

        # Last share ensures sum equals secret
        final_share = secret - running_sum
        shares.append(final_share)

        return shares

    def reconstruct(self, shares: List[torch.Tensor]) -> torch.Tensor:
        """Reconstruct from shares."""
        if len(shares) < self.threshold:
            raise ValueError("Not enough shares")

        return sum(shares)


class MPCServer:
    """MPC computation server."""

    def __init__(self, server_id: int, config: MPCConfig):
        self.server_id = server_id
        self.config = config
        self.shares: Dict[str, List[torch.Tensor]] = {}

    def receive_share(self, key: str, share: torch.Tensor) -> None:
        """Receive a share from client."""
        if key not in self.shares:
            self.shares[key] = []
        self.shares[key].append(share)

    def local_aggregate(self, key: str) -> torch.Tensor:
        """Aggregate shares locally."""
        if key not in self.shares:
            return None

        return sum(self.shares[key])

    def clear(self) -> None:
        """Clear stored shares."""
        self.shares.clear()


class MPCCoordinator:
    """Coordinate MPC across servers."""

    def __init__(self, servers: List[MPCServer], sharing: SecretSharing):
        self.servers = servers
        self.sharing = sharing

    def distribute_secret(self, key: str, secret: torch.Tensor) -> None:
        """Distribute secret to servers."""
        shares = self.sharing.share(secret)

        for server, share in zip(self.servers, shares):
            server.receive_share(key, share)

    def secure_aggregate(self, key: str) -> torch.Tensor:
        """Securely aggregate across servers."""
        local_sums = [s.local_aggregate(key) for s in self.servers]

        # Clear servers
        for s in self.servers:
            s.clear()

        return self.sharing.reconstruct(local_sums)


class MPCDataset(Dataset):
    def __init__(self, n: int = 200, dim: int = 32, classes: int = 10, seed: int = 0):
        np.random.seed(seed)
        self.x = torch.randn(n, dim, dtype=torch.float32)
        self.y = torch.randint(0, classes, (n,), dtype=torch.long)
        for i in range(n):
            self.x[i, self.y[i].item() % dim] += 2.0

    def __len__(self): return len(self.y)
    def __getitem__(self, idx): return self.x[idx], self.y[idx]


class MPCModel(nn.Module):
    def __init__(self, config: MPCConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.num_classes)
        )

    def forward(self, x): return self.net(x)


class MPCClient:
    """Client using MPC."""

    def __init__(self, client_id: int, dataset: MPCDataset, config: MPCConfig, coordinator: MPCCoordinator):
        self.client_id = client_id
        self.dataset = dataset
        self.config = config
        self.coordinator = coordinator

    def train(self, model: nn.Module) -> None:
        local = copy.deepcopy(model)
        optimizer = torch.optim.SGD(local.parameters(), lr=self.config.learning_rate)
        loader = DataLoader(self.dataset, batch_size=self.config.batch_size, shuffle=True)

        local.train()
        for _ in range(self.config.local_epochs):
            for x, y in loader:
                optimizer.zero_grad()
                loss = F.cross_entropy(local(x), y)
                loss.backward()
                optimizer.step()

        # Distribute model update via MPC
        for k, v in local.state_dict().items():
            key = f"{k}_{self.client_id}"
            self.coordinator.distribute_secret(key, v.cpu())


class MPCFLServer:
    """FL server with MPC."""

    def __init__(
        self,
        model: nn.Module,
        clients: List[MPCClient],
        test_data: MPCDataset,
        config: MPCConfig,
        coordinator: MPCCoordinator
    ):
        self.model = model
        self.clients = clients
        self.test_data = test_data
        self.config = config
        self.coordinator = coordinator
        self.history: List[Dict] = []

    def aggregate(self, selected_ids: List[int]) -> None:
        """Aggregate using MPC."""
        keys = list(self.model.state_dict().keys())

        aggregated = {}
        for k in keys:
            all_values = []
            for cid in selected_ids:
                key = f"{k}_{cid}"
                value = self.coordinator.secure_aggregate(key)
                all_values.append(value)

            aggregated[k] = sum(all_values) / len(all_values)

        self.model.load_state_dict(aggregated)

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
        logger.info(f"Starting MPC-FL with {self.config.num_servers} servers")

        for round_num in range(self.config.num_rounds):
            n = min(self.config.clients_per_round, len(self.clients))
            indices = np.random.choice(len(self.clients), n, replace=False)
            selected = [self.clients[i] for i in indices]

            for c in selected:
                c.train(self.model)

            self.aggregate(list(indices))

            metrics = self.evaluate()

            record = {"round": round_num, **metrics}
            self.history.append(record)

            if (round_num + 1) % 10 == 0:
                logger.info(f"Round {round_num + 1}: acc={metrics['accuracy']:.4f}")

        return self.history


def main():
    print("=" * 60)
    print("Tutorial 100: FL Multi-Party Computation")
    print("=" * 60)

    config = MPCConfig()
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    # Setup MPC infrastructure
    servers = [MPCServer(i, config) for i in range(config.num_servers)]
    sharing = SecretSharing(config.num_servers, config.threshold)
    coordinator = MPCCoordinator(servers, sharing)

    clients = [MPCClient(i, MPCDataset(seed=config.seed + i), config, coordinator) for i in range(config.num_clients)]
    test_data = MPCDataset(seed=999)
    model = MPCModel(config)

    server = MPCFLServer(model, clients, test_data, config, coordinator)
    history = server.train()

    print("\n" + "=" * 60)
    print("MPC Training Complete")
    print(f"Final accuracy: {history[-1]['accuracy']:.4f}")
    print(f"MPC servers: {config.num_servers}")
    print(f"Threshold: {config.threshold}")
    print("=" * 60)


if __name__ == "__main__":
    main()
```

---

## Key Insights

### MPC Best Practices

1. **Server selection**: Choose independent parties
2. **Communication**: Minimize rounds
3. **Threshold**: Balance security and availability
4. **Verification**: Add integrity checks

---

## Exercises

1. **Exercise 1**: Implement Shamir sharing
2. **Exercise 2**: Add secure multiplication
3. **Exercise 3**: Design malicious security
4. **Exercise 4**: Optimize communication

---

## References

1. Yao, A.C. (1982). Protocols for secure computations. In *FOCS*.
2. Goldreich, O., et al. (1987). How to play any mental game. In *STOC*.
3. Mohassel, P., & Zhang, Y. (2017). SecureML. In *S&P*.

---

*Copyright 2026 Olaf Yunus Laitinen Imanov and Contributors. Released under EUPL 1.2.*
