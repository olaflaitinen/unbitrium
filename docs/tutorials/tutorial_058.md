# Tutorial 058: FL Split Learning

---

## Metadata

| Property | Value |
|----------|-------|
| **Tutorial ID** | 058 |
| **Title** | Federated Learning Split Learning |
| **Category** | Advanced Architectures |
| **Difficulty** | Expert |
| **Duration** | 120 minutes |
| **Prerequisites** | Tutorial 001-057 |
| **Author** | Unbitrium Contributors |
| **Last Updated** | January 2026 |

---

## Learning Objectives

By completing this tutorial, you will be able to:

1. **Understand** split learning concepts
2. **Implement** client-server model splitting
3. **Design** privacy-preserving splits
4. **Analyze** communication trade-offs
5. **Deploy** split FL systems

---

## Prerequisites

Before starting this tutorial, ensure you have:

- Completed tutorials 001-057
- Understanding of FL fundamentals
- Knowledge of neural network layers
- Familiarity with backpropagation

---

## Background and Theory

### Split Learning Architecture

```
Split Learning:
├── Client Side
│   ├── Bottom layers
│   ├── Input data
│   └── Forward to cut
├── Server Side
│   ├── Top layers
│   ├── Loss computation
│   └── Backward to cut
├── Communication
│   ├── Activations (forward)
│   ├── Gradients (backward)
│   └── Cut layer
└── Variants
    ├── U-shaped split
    ├── Multi-cut
    └── SplitFed
```

---

## Implementation Code

```python
#!/usr/bin/env python3
"""
Tutorial 058: FL Split Learning

This module implements split learning where model
is divided between client and server.

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
class SplitConfig:
    """Split learning configuration."""

    num_rounds: int = 50
    num_clients: int = 10
    clients_per_round: int = 5

    input_dim: int = 32
    hidden_dim: int = 64
    cut_dim: int = 32
    num_classes: int = 10

    learning_rate: float = 0.01
    batch_size: int = 32
    local_epochs: int = 3

    seed: int = 42


class ClientModel(nn.Module):
    """Client-side model (bottom layers)."""

    def __init__(self, config: SplitConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.cut_dim),
            nn.ReLU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ServerModel(nn.Module):
    """Server-side model (top layers)."""

    def __init__(self, config: SplitConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.cut_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SplitDataset(Dataset):
    def __init__(self, n: int = 200, dim: int = 32, classes: int = 10, seed: int = 0):
        np.random.seed(seed)
        self.x = torch.randn(n, dim, dtype=torch.float32)
        self.y = torch.randint(0, classes, (n,), dtype=torch.long)
        for i in range(n):
            self.x[i, self.y[i].item() % dim] += 2.0

    def __len__(self): return len(self.y)
    def __getitem__(self, idx): return self.x[idx], self.y[idx]


class SplitClient:
    """Client in split learning."""

    def __init__(
        self,
        client_id: int,
        dataset: SplitDataset,
        config: SplitConfig
    ):
        self.client_id = client_id
        self.dataset = dataset
        self.config = config

        self.model = ClientModel(config)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass to cut layer."""
        self.model.train()
        return self.model(x)

    def backward(self, grad: torch.Tensor, activations: torch.Tensor) -> None:
        """Backward pass from cut layer gradient."""
        self.optimizer.zero_grad()
        activations.backward(grad)
        self.optimizer.step()

    def get_state(self) -> Dict[str, torch.Tensor]:
        return {k: v.cpu().clone() for k, v in self.model.state_dict().items()}

    def set_state(self, state: Dict[str, torch.Tensor]) -> None:
        self.model.load_state_dict(state)


class SplitServer:
    """Server in split learning."""

    def __init__(self, config: SplitConfig):
        self.config = config
        self.model = ServerModel(config)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate)

    def forward_backward(
        self,
        activations: torch.Tensor,
        labels: torch.Tensor
    ) -> Tuple[torch.Tensor, float]:
        """Forward and backward pass on server."""
        self.model.train()

        # Need gradients for activations
        activations = activations.detach().requires_grad_(True)

        self.optimizer.zero_grad()
        output = self.model(activations)
        loss = F.cross_entropy(output, labels)
        loss.backward()
        self.optimizer.step()

        # Return gradient for cut layer
        return activations.grad, loss.item()

    def get_state(self) -> Dict[str, torch.Tensor]:
        return {k: v.cpu().clone() for k, v in self.model.state_dict().items()}


class SplitCoordinator:
    """Coordinate split learning."""

    def __init__(
        self,
        clients: List[SplitClient],
        server: SplitServer,
        test_data: SplitDataset,
        config: SplitConfig
    ):
        self.clients = clients
        self.server = server
        self.test_data = test_data
        self.config = config
        self.history: List[Dict] = []

    def train_iteration(
        self,
        client: SplitClient,
        x: torch.Tensor,
        y: torch.Tensor
    ) -> float:
        """One training iteration."""
        # Client forward
        activations = client.forward(x)

        # Detach for transfer
        activations_detached = activations.detach().requires_grad_(True)

        # Server forward/backward
        grad, loss = self.server.forward_backward(activations_detached, y)

        # Client backward
        client.backward(grad, activations)

        return loss

    def aggregate_clients(self) -> None:
        """Aggregate client models."""
        n = len(self.clients)
        new_state = {}

        for key in self.clients[0].get_state().keys():
            new_state[key] = sum(c.get_state()[key] for c in self.clients) / n

        for client in self.clients:
            client.set_state(new_state)

    def evaluate(self) -> Dict[str, float]:
        """Evaluate combined model."""
        client = self.clients[0]
        client.model.eval()
        self.server.model.eval()

        loader = DataLoader(self.test_data, batch_size=64)
        correct, total = 0, 0

        with torch.no_grad():
            for x, y in loader:
                activations = client.model(x)
                output = self.server.model(activations)
                pred = output.argmax(dim=1)
                correct += (pred == y).sum().item()
                total += len(y)

        return {"accuracy": correct / total}

    def train(self) -> List[Dict]:
        logger.info("Starting split learning")

        for round_num in range(self.config.num_rounds):
            n = min(self.config.clients_per_round, len(self.clients))
            indices = np.random.choice(len(self.clients), n, replace=False)
            selected = [self.clients[i] for i in indices]

            for client in selected:
                loader = DataLoader(client.dataset, batch_size=self.config.batch_size, shuffle=True)

                for _ in range(self.config.local_epochs):
                    for x, y in loader:
                        self.train_iteration(client, x, y)

            self.aggregate_clients()

            metrics = self.evaluate()

            record = {"round": round_num, **metrics}
            self.history.append(record)

            if (round_num + 1) % 10 == 0:
                logger.info(f"Round {round_num + 1}: acc={metrics['accuracy']:.4f}")

        return self.history


def main():
    print("=" * 60)
    print("Tutorial 058: FL Split Learning")
    print("=" * 60)

    config = SplitConfig()
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    clients = [
        SplitClient(i, SplitDataset(seed=config.seed + i), config)
        for i in range(config.num_clients)
    ]
    server = SplitServer(config)
    test_data = SplitDataset(seed=999)

    coordinator = SplitCoordinator(clients, server, test_data, config)
    history = coordinator.train()

    # Communication analysis
    client_params = sum(p.numel() for p in clients[0].model.parameters())
    server_params = sum(p.numel() for p in server.model.parameters())

    print("\n" + "=" * 60)
    print("Split Learning Complete")
    print(f"Client params: {client_params:,}")
    print(f"Server params: {server_params:,}")
    print(f"Final accuracy: {history[-1]['accuracy']:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
```

---

## Key Insights

### Split Learning Trade-offs

1. **Privacy**: Raw data stays on client
2. **Communication**: Activations vs gradients
3. **Compute**: Offload to server
4. **Flexibility**: Different cut points

---

## Exercises

1. **Exercise 1**: Implement U-shaped split
2. **Exercise 2**: Add differential privacy
3. **Exercise 3**: Design parallel split
4. **Exercise 4**: Combine with FL

---

## References

1. Gupta, O., & Raskar, R. (2018). Distributed learning of DNNs: A split learning approach. *arXiv*.
2. Thapa, C., et al. (2022). SplitFed: When federated learning meets split learning. In *AAAI*.
3. Vepakomma, P., et al. (2018). Split learning for health. *arXiv*.

---

*Copyright 2026 Olaf Yunus Laitinen Imanov and Contributors. Released under EUPL 1.2.*
