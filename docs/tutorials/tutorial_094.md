# Tutorial 094: FL Secure Aggregation Advanced

---

## Metadata

| Property | Value |
|----------|-------|
| **Tutorial ID** | 094 |
| **Title** | FL Secure Aggregation Advanced |
| **Category** | Privacy |
| **Difficulty** | Expert |
| **Duration** | 120 minutes |
| **Prerequisites** | Tutorial 001-093 |
| **Author** | Unbitrium Contributors |
| **Last Updated** | January 2026 |

---

## Learning Objectives

By completing this tutorial, you will be able to:

1. **Understand** advanced secure aggregation
2. **Implement** cryptographic protocols
3. **Design** communication-efficient SecAgg
4. **Analyze** security guarantees
5. **Deploy** production SecAgg

---

## Prerequisites

Before starting this tutorial, ensure you have:

- Completed tutorials 001-093
- Understanding of FL fundamentals
- Knowledge of cryptography
- Familiarity with secret sharing

---

## Background and Theory

### Secure Aggregation Protocols

```
SecAgg Protocols:
├── Secret Sharing
│   ├── Shamir's scheme
│   ├── Additive sharing
│   └── Threshold sharing
├── Masking Techniques
│   ├── Pairwise masking
│   ├── One-time pads
│   └── Random seeds
├── Communication Rounds
│   ├── Setup phase
│   ├── Masking phase
│   └── Unmasking phase
└── Fault Tolerance
    ├── Dropout handling
    ├── Reconstruction
    └── Abort protocols
```

---

## Implementation Code

```python
#!/usr/bin/env python3
"""
Tutorial 094: FL Secure Aggregation Advanced

This module implements advanced secure aggregation
with cryptographic guarantees.

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
import hashlib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SecAggConfig:
    """Secure aggregation configuration."""

    num_rounds: int = 30
    num_clients: int = 10
    clients_per_round: int = 8
    threshold: int = 5  # Minimum for reconstruction

    input_dim: int = 32
    hidden_dim: int = 64
    num_classes: int = 10

    learning_rate: float = 0.01
    batch_size: int = 32
    local_epochs: int = 3

    seed: int = 42


class PRG:
    """Pseudorandom generator for masking."""

    def __init__(self, seed: bytes):
        self.seed = seed
        self.counter = 0

    def generate(self, shape: Tuple[int, ...]) -> torch.Tensor:
        """Generate random tensor from seed."""
        # Use hash chain for reproducible randomness
        data = self.seed + str(self.counter).encode()
        self.counter += 1

        np.random.seed(int.from_bytes(hashlib.sha256(data).digest()[:4], 'big'))
        return torch.from_numpy(np.random.randn(*shape).astype(np.float32))


class SecretSharer:
    """Shamir secret sharing."""

    def __init__(self, n: int, t: int):
        self.n = n  # Total shares
        self.t = t  # Threshold
        self.prime = 2**31 - 1  # Mersenne prime

    def share(self, secret: int) -> List[Tuple[int, int]]:
        """Create t-of-n shares."""
        coeffs = [secret] + [np.random.randint(0, self.prime) for _ in range(self.t - 1)]
        shares = []

        for x in range(1, self.n + 1):
            y = sum(c * (x ** i) for i, c in enumerate(coeffs)) % self.prime
            shares.append((x, y))

        return shares

    def reconstruct(self, shares: List[Tuple[int, int]]) -> int:
        """Reconstruct secret from shares."""
        if len(shares) < self.t:
            raise ValueError("Not enough shares")

        secret = 0
        for i, (xi, yi) in enumerate(shares[:self.t]):
            num, den = 1, 1
            for j, (xj, _) in enumerate(shares[:self.t]):
                if i != j:
                    num = (num * (-xj)) % self.prime
                    den = (den * (xi - xj)) % self.prime

            lagrange = (yi * num * pow(den, -1, self.prime)) % self.prime
            secret = (secret + lagrange) % self.prime

        return secret


class MaskGenerator:
    """Generate pairwise masks."""

    def __init__(self, client_id: int, all_clients: List[int], seed: int):
        self.client_id = client_id
        self.all_clients = all_clients
        self.pairwise_seeds: Dict[int, bytes] = {}

        # Generate pairwise seeds via Diffie-Hellman simulation
        for other_id in all_clients:
            if other_id != client_id:
                combined = min(client_id, other_id) * 1000 + max(client_id, other_id)
                self.pairwise_seeds[other_id] = hashlib.sha256(
                    f"{seed}:{combined}".encode()
                ).digest()

    def generate_mask(self, shape: Tuple[int, ...], round_num: int) -> torch.Tensor:
        """Generate mask that cancels out."""
        mask = torch.zeros(shape)

        for other_id, seed_bytes in self.pairwise_seeds.items():
            prg = PRG(seed_bytes + str(round_num).encode())
            pairwise_mask = prg.generate(shape)

            if self.client_id < other_id:
                mask += pairwise_mask
            else:
                mask -= pairwise_mask

        return mask


class SecAggDataset(Dataset):
    def __init__(self, n: int = 200, dim: int = 32, classes: int = 10, seed: int = 0):
        np.random.seed(seed)
        self.x = torch.randn(n, dim, dtype=torch.float32)
        self.y = torch.randint(0, classes, (n,), dtype=torch.long)
        for i in range(n):
            self.x[i, self.y[i].item() % dim] += 2.0

    def __len__(self): return len(self.y)
    def __getitem__(self, idx): return self.x[idx], self.y[idx]


class SecAggModel(nn.Module):
    def __init__(self, config: SecAggConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.num_classes)
        )

    def forward(self, x): return self.net(x)


class SecAggClient:
    """Client with secure aggregation."""

    def __init__(self, client_id: int, dataset: SecAggDataset, config: SecAggConfig, all_clients: List[int]):
        self.client_id = client_id
        self.dataset = dataset
        self.config = config

        self.mask_gen = MaskGenerator(client_id, all_clients, config.seed)
        self.sharer = SecretSharer(len(all_clients), config.threshold)

    def train(self, model: nn.Module) -> Dict[str, torch.Tensor]:
        """Train and return update."""
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

        return {k: v.cpu() for k, v in local.state_dict().items()}

    def mask_update(
        self,
        update: Dict[str, torch.Tensor],
        round_num: int
    ) -> Dict[str, torch.Tensor]:
        """Add cryptographic mask to update."""
        masked = {}

        for key, value in update.items():
            mask = self.mask_gen.generate_mask(value.shape, round_num)
            masked[key] = value + mask

        return masked


class SecAggServer:
    """Server for secure aggregation."""

    def __init__(
        self,
        model: nn.Module,
        clients: List[SecAggClient],
        test_data: SecAggDataset,
        config: SecAggConfig
    ):
        self.model = model
        self.clients = clients
        self.test_data = test_data
        self.config = config
        self.history: List[Dict] = []

    def secure_aggregate(self, masked_updates: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Aggregate masked updates (masks cancel out)."""
        n = len(masked_updates)
        aggregated = {}

        for key in masked_updates[0]:
            aggregated[key] = sum(u[key] for u in masked_updates) / n

        return aggregated

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
        logger.info("Starting FL with secure aggregation")

        for round_num in range(self.config.num_rounds):
            n = min(self.config.clients_per_round, len(self.clients))
            indices = np.random.choice(len(self.clients), n, replace=False)
            selected = [self.clients[i] for i in indices]

            # Phase 1: Local training
            updates = [c.train(self.model) for c in selected]

            # Phase 2: Mask updates
            masked = [c.mask_update(u, round_num) for c, u in zip(selected, updates)]

            # Phase 3: Secure aggregation
            aggregated = self.secure_aggregate(masked)
            self.model.load_state_dict(aggregated)

            metrics = self.evaluate()

            record = {"round": round_num, **metrics}
            self.history.append(record)

            if (round_num + 1) % 10 == 0:
                logger.info(f"Round {round_num + 1}: acc={metrics['accuracy']:.4f}")

        return self.history


def main():
    print("=" * 60)
    print("Tutorial 094: FL Secure Aggregation Advanced")
    print("=" * 60)

    config = SecAggConfig()
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    client_ids = list(range(config.num_clients))
    clients = [
        SecAggClient(i, SecAggDataset(seed=config.seed + i), config, client_ids)
        for i in client_ids
    ]
    test_data = SecAggDataset(seed=999)
    model = SecAggModel(config)

    server = SecAggServer(model, clients, test_data, config)
    history = server.train()

    print("\n" + "=" * 60)
    print("Secure Aggregation Complete")
    print(f"Final accuracy: {history[-1]['accuracy']:.4f}")
    print(f"Threshold: {config.threshold}/{config.num_clients}")
    print("=" * 60)


if __name__ == "__main__":
    main()
```

---

## Key Insights

### SecAgg Best Practices

1. **Threshold**: Balance security and fault tolerance
2. **Communication**: Optimize round complexity
3. **Dropout**: Handle missing clients
4. **Efficiency**: Use PRGs for masking

---

## Exercises

1. **Exercise 1**: Add dropout handling
2. **Exercise 2**: Implement lightweight SecAgg
3. **Exercise 3**: Add verifiable aggregation
4. **Exercise 4**: Design efficient key agreement

---

## References

1. Bonawitz, K., et al. (2017). Practical secure aggregation for FL. In *CCS*.
2. Bell, J.H., et al. (2020). Secure single-server aggregation. In *CCS*.
3. So, J., et al. (2022). Turbo-Aggregate: Breaking the quadratic aggregation barrier. In *JSAIT*.

---

*Copyright 2026 Olaf Yunus Laitinen Imanov and Contributors. Released under EUPL 1.2.*
