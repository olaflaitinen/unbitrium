# Tutorial 167: FL Vertical Partitioning

---

## Metadata

| Property | Value |
|----------|-------|
| **Tutorial ID** | 167 |
| **Title** | FL Vertical Partitioning |
| **Category** | Advanced Topics |
| **Difficulty** | Expert |
| **Duration** | 120 minutes |
| **Prerequisites** | Tutorial 001-166 |
| **Author** | Unbitrium Contributors |
| **Last Updated** | January 2026 |

---

## Learning Objectives

By completing this tutorial, you will be able to:

1. **Understand** vertical FL
2. **Implement** split feature learning
3. **Design** secure aggregation
4. **Analyze** privacy guarantees
5. **Deploy** vertical FL systems

---

## Background and Theory

### Vertical FL Architecture

```
Vertical FL:
├── Data Partitioning
│   ├── Same samples
│   ├── Different features
│   └── Entity alignment
├── Computation
│   ├── Split neural networks
│   ├── Secure aggregation
│   └── Gradient exchange
├── Coordination
│   ├── Sample alignment
│   ├── Label party
│   └── Feature parties
└── Applications
    ├── Cross-org data
    ├── Feature enrichment
    └── Privacy-preserving joins
```

---

## Implementation Code

```python
#!/usr/bin/env python3
"""
Tutorial 167: FL Vertical Partitioning

This module implements vertical federated learning
where parties have different features for same samples.

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
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class VerticalConfig:
    """Vertical FL configuration."""

    num_epochs: int = 50
    num_parties: int = 3  # Feature parties + label party

    total_features: int = 60
    features_per_party: int = 20
    hidden_dim: int = 32
    num_classes: int = 10

    learning_rate: float = 0.01
    batch_size: int = 32

    seed: int = 42


class VerticalDataset:
    """Dataset split vertically across parties."""

    def __init__(
        self,
        n_samples: int = 1000,
        total_features: int = 60,
        num_parties: int = 3,
        num_classes: int = 10,
        seed: int = 0
    ):
        np.random.seed(seed)

        self.n_samples = n_samples
        self.num_parties = num_parties

        # Generate full dataset
        self.x_full = torch.randn(n_samples, total_features, dtype=torch.float32)
        self.y = torch.randint(0, num_classes, (n_samples,), dtype=torch.long)

        # Add class signal
        for i in range(n_samples):
            self.x_full[i, self.y[i].item() % total_features] += 2.0

        # Split features vertically
        features_per_party = total_features // num_parties
        self.party_data = {}

        for p in range(num_parties):
            start = p * features_per_party
            end = start + features_per_party
            self.party_data[p] = self.x_full[:, start:end]

    def get_party_batch(self, party_id: int, indices: List[int]) -> torch.Tensor:
        """Get batch for specific party."""
        return self.party_data[party_id][indices]

    def get_labels(self, indices: List[int]) -> torch.Tensor:
        """Get labels (only label party has this)."""
        return self.y[indices]


class PartyModel(nn.Module):
    """Local model for a feature party."""

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AggregatorModel(nn.Module):
    """Model on label party that aggregates embeddings."""

    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FeatureParty:
    """A party holding partial features."""

    def __init__(
        self,
        party_id: int,
        input_dim: int,
        embedding_dim: int,
        learning_rate: float
    ):
        self.party_id = party_id
        self.model = PartyModel(input_dim, embedding_dim)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute local embedding."""
        self.model.train()
        return self.model(x)

    def backward(self, grad: torch.Tensor, embedding: torch.Tensor) -> None:
        """Backward pass from aggregator gradient."""
        self.optimizer.zero_grad()
        embedding.backward(grad)
        self.optimizer.step()


class LabelParty:
    """Party holding labels and aggregator."""

    def __init__(
        self,
        total_embedding_dim: int,
        num_classes: int,
        learning_rate: float
    ):
        self.model = AggregatorModel(total_embedding_dim, num_classes)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

    def forward_backward(
        self,
        embeddings: List[torch.Tensor],
        labels: torch.Tensor
    ) -> Tuple[float, List[torch.Tensor]]:
        """Forward and backward, return loss and gradients."""
        # Concatenate embeddings
        combined = torch.cat(embeddings, dim=1)
        combined.requires_grad_(True)

        self.optimizer.zero_grad()
        output = self.model(combined)
        loss = F.cross_entropy(output, labels)
        loss.backward()
        self.optimizer.step()

        # Split gradient for each party
        grad = combined.grad
        grads = []
        offset = 0
        for emb in embeddings:
            dim = emb.shape[1]
            grads.append(grad[:, offset:offset + dim])
            offset += dim

        return loss.item(), grads

    def evaluate(self, embeddings: List[torch.Tensor], labels: torch.Tensor) -> float:
        """Evaluate accuracy."""
        self.model.eval()
        combined = torch.cat(embeddings, dim=1)
        with torch.no_grad():
            output = self.model(combined)
            pred = output.argmax(dim=1)
            accuracy = (pred == labels).float().mean().item()
        return accuracy


class VerticalFLCoordinator:
    """Coordinate vertical FL training."""

    def __init__(self, config: VerticalConfig):
        self.config = config

        # Create dataset
        self.dataset = VerticalDataset(
            n_samples=1000,
            total_features=config.total_features,
            num_parties=config.num_parties,
            num_classes=config.num_classes,
            seed=config.seed
        )

        # Create feature parties
        embedding_dim = config.hidden_dim
        self.feature_parties = [
            FeatureParty(i, config.features_per_party, embedding_dim, config.learning_rate)
            for i in range(config.num_parties)
        ]

        # Create label party
        total_embedding = embedding_dim * config.num_parties
        self.label_party = LabelParty(total_embedding, config.num_classes, config.learning_rate)

        self.history: List[Dict] = []

    def train(self) -> List[Dict]:
        logger.info(f"Starting vertical FL with {self.config.num_parties} parties")

        n_samples = self.dataset.n_samples
        n_batches = n_samples // self.config.batch_size

        for epoch in range(self.config.num_epochs):
            epoch_loss = 0.0

            indices = np.random.permutation(n_samples)

            for b in range(n_batches):
                batch_idx = list(indices[b * self.config.batch_size:(b + 1) * self.config.batch_size])

                # Each party computes embeddings
                embeddings = []
                for i, party in enumerate(self.feature_parties):
                    x = self.dataset.get_party_batch(i, batch_idx)
                    emb = party.forward(x)
                    embeddings.append(emb)

                labels = self.dataset.get_labels(batch_idx)

                # Label party aggregates and computes loss
                loss, grads = self.label_party.forward_backward(embeddings, labels)
                epoch_loss += loss

                # Send gradients back to feature parties
                for party, grad, emb in zip(self.feature_parties, grads, embeddings):
                    party.backward(grad, emb)

            # Evaluate
            all_idx = list(range(n_samples))
            embeddings = [party.forward(self.dataset.get_party_batch(i, all_idx))
                         for i, party in enumerate(self.feature_parties)]
            labels = self.dataset.get_labels(all_idx)
            accuracy = self.label_party.evaluate(embeddings, labels)

            record = {"epoch": epoch, "loss": epoch_loss / n_batches, "accuracy": accuracy}
            self.history.append(record)

            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch + 1}: loss={epoch_loss / n_batches:.4f}, acc={accuracy:.4f}")

        return self.history


def main():
    print("=" * 60)
    print("Tutorial 167: FL Vertical Partitioning")
    print("=" * 60)

    config = VerticalConfig()
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    coordinator = VerticalFLCoordinator(config)
    history = coordinator.train()

    print("\n" + "=" * 60)
    print("Vertical FL Complete")
    print(f"Final accuracy: {history[-1]['accuracy']:.4f}")
    print(f"Parties: {config.num_parties}")
    print("=" * 60)


if __name__ == "__main__":
    main()
```

---

## Key Insights

### Vertical FL Best Practices

1. **Entity alignment**: Same samples across parties
2. **Gradient security**: Protect shared gradients
3. **Label privacy**: Only label party sees labels
4. **Communication**: Minimize embedding exchange

---

## Exercises

1. **Exercise 1**: Add secure gradient exchange
2. **Exercise 2**: Implement label DP
3. **Exercise 3**: Design sample alignment
4. **Exercise 4**: Add heterogeneous models

---

## References

1. Liu, Y., et al. (2020). A vertical FL framework. *arXiv*.
2. Yang, Q., et al. (2019). Federated machine learning. *TIST*.
3. Cheng, K., et al. (2020). SecureBoost: A lossless vertical FL framework. *arXiv*.

---

*Copyright 2026 Olaf Yunus Laitinen Imanov and Contributors. Released under EUPL 1.2.*
