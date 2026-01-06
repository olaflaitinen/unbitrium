# Tutorial 112: FL with GNNs

---

## Metadata

| Property | Value |
|----------|-------|
| **Tutorial ID** | 112 |
| **Title** | Federated Learning with Graph Neural Networks |
| **Category** | Advanced Architectures |
| **Difficulty** | Expert |
| **Duration** | 120 minutes |
| **Prerequisites** | Tutorial 001-111 |
| **Author** | Unbitrium Contributors |
| **Last Updated** | January 2026 |

---

## Learning Objectives

By completing this tutorial, you will be able to:

1. **Understand** FL challenges for graph data
2. **Implement** federated GNN training
3. **Design** graph partitioning strategies
4. **Analyze** cross-subgraph learning
5. **Deploy** federated graph analytics

---

## Prerequisites

Before starting this tutorial, ensure you have:

- Completed tutorials 001-111
- Understanding of FL fundamentals
- Knowledge of graph neural networks
- Familiarity with message passing

---

## Background and Theory

### GNN Challenges in FL

Graph data presents unique FL challenges:
- Graphs span multiple clients
- Node features are private
- Message passing crosses boundaries
- Graph structure may be sensitive

### Federated GNN Architectures

```
Federated GNN Patterns:
├── Subgraph FL
│   ├── Each client owns subgraph
│   ├── No cross-client edges
│   └── Independent training
├── Cross-Client Graphs
│   ├── Graph spans clients
│   ├── Privacy-preserving aggregation
│   └── Secure message passing
├── Knowledge Distillation
│   ├── Local GNN training
│   ├── Knowledge transfer
│   └── Global model
└── Personalized
    ├── Client-specific GNNs
    ├── Shared backbone
    └── Personal heads
```

### GNN Aggregation Patterns

| Pattern | Graph Type | Privacy |
|---------|------------|---------|
| Node-level | Connected | Medium |
| Subgraph | Disconnected | High |
| Graph-level | Multiple graphs | High |

---

## Implementation Code

```python
#!/usr/bin/env python3
"""
Tutorial 112: Federated Learning with GNNs

This module implements federated GNN training for
graph-structured data across distributed clients.

Copyright 2026 Olaf Yunus Laitinen Imanov and Contributors
Released under EUPL 1.2
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import copy
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class GNNConfig:
    """Configuration for federated GNN."""

    num_rounds: int = 50
    num_clients: int = 10
    clients_per_round: int = 5

    # GNN parameters
    node_features: int = 16
    hidden_dim: int = 32
    num_classes: int = 4
    num_layers: int = 2

    # Graph parameters
    nodes_per_client: int = 50
    edge_density: float = 0.1

    learning_rate: float = 0.01
    local_epochs: int = 5

    seed: int = 42


class GraphData:
    """Graph data structure."""

    def __init__(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        y: torch.Tensor
    ):
        self.x = x  # Node features [num_nodes, features]
        self.edge_index = edge_index  # [2, num_edges]
        self.y = y  # Node labels [num_nodes]
        self.num_nodes = x.size(0)

    @staticmethod
    def random_graph(
        num_nodes: int,
        num_features: int,
        num_classes: int,
        edge_density: float,
        seed: int = 0
    ) -> 'GraphData':
        """Generate random graph."""
        np.random.seed(seed)
        torch.manual_seed(seed)

        # Node features
        x = torch.randn(num_nodes, num_features)

        # Labels with community structure
        y = torch.randint(0, num_classes, (num_nodes,))

        # Add class signal to features
        for i in range(num_nodes):
            x[i, y[i].item() % num_features] += 1.0

        # Generate edges (preferential to same class)
        edges = []
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                prob = edge_density
                if y[i] == y[j]:
                    prob *= 2  # Higher for same class

                if np.random.random() < prob:
                    edges.append([i, j])
                    edges.append([j, i])

        if len(edges) == 0:
            # Ensure at least some edges
            edges = [[0, 1], [1, 0]]

        edge_index = torch.tensor(edges, dtype=torch.long).t()

        return GraphData(x, edge_index, y)


class GCNLayer(nn.Module):
    """Graph Convolutional Layer."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor
    ) -> torch.Tensor:
        """Forward with message passing."""
        num_nodes = x.size(0)

        # Compute degree for normalization
        row, col = edge_index
        deg = torch.zeros(num_nodes, dtype=x.dtype)
        deg.scatter_add_(0, row, torch.ones(row.size(0)))
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        # Normalize
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # Aggregate
        aggr = torch.zeros_like(x)
        for i in range(edge_index.size(1)):
            src, dst = edge_index[:, i]
            aggr[dst] += norm[i] * x[src]

        # Transform
        out = self.linear(aggr + x)

        return out


class FedGNN(nn.Module):
    """Federated Graph Neural Network."""

    def __init__(self, config: GNNConfig):
        super().__init__()

        self.layers = nn.ModuleList()

        # Input layer
        self.layers.append(GCNLayer(config.node_features, config.hidden_dim))

        # Hidden layers
        for _ in range(config.num_layers - 1):
            self.layers.append(GCNLayer(config.hidden_dim, config.hidden_dim))

        # Output
        self.classifier = nn.Linear(config.hidden_dim, config.num_classes)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass."""
        for layer in self.layers:
            x = F.relu(layer(x, edge_index))
            x = F.dropout(x, p=0.2, training=self.training)

        return self.classifier(x)


class GNNClient:
    """FL client with local subgraph."""

    def __init__(
        self,
        client_id: int,
        graph: GraphData,
        config: GNNConfig
    ):
        self.client_id = client_id
        self.graph = graph
        self.config = config

    def train(self, model: nn.Module) -> Dict[str, Any]:
        """Train on local subgraph."""
        local = copy.deepcopy(model)
        optimizer = torch.optim.Adam(
            local.parameters(),
            lr=self.config.learning_rate
        )

        local.train()
        total_loss = 0.0
        num_epochs = 0

        for _ in range(self.config.local_epochs):
            optimizer.zero_grad()

            out = local(self.graph.x, self.graph.edge_index)
            loss = F.cross_entropy(out, self.graph.y)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(local.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            num_epochs += 1

        return {
            "state_dict": {k: v.cpu() for k, v in local.state_dict().items()},
            "num_samples": self.graph.num_nodes,
            "avg_loss": total_loss / num_epochs,
            "client_id": self.client_id
        }

    def evaluate(self, model: nn.Module) -> Dict[str, float]:
        """Evaluate on local graph."""
        model.eval()

        with torch.no_grad():
            out = model(self.graph.x, self.graph.edge_index)
            pred = out.argmax(dim=1)
            correct = (pred == self.graph.y).sum().item()
            total = self.graph.num_nodes

        return {
            "accuracy": correct / total,
            "num_nodes": total
        }


class GNNServer:
    """Server for federated GNN."""

    def __init__(
        self,
        model: nn.Module,
        clients: List[GNNClient],
        test_graph: GraphData,
        config: GNNConfig
    ):
        self.model = model
        self.clients = clients
        self.test_graph = test_graph
        self.config = config
        self.history: List[Dict] = []

    def aggregate(self, updates: List[Dict]) -> None:
        """Aggregate GNN updates."""
        if not updates:
            return

        total_nodes = sum(u["num_samples"] for u in updates)
        new_state = {}

        for key in updates[0]["state_dict"]:
            new_state[key] = sum(
                (u["num_samples"] / total_nodes) * u["state_dict"][key].float()
                for u in updates
            )

        self.model.load_state_dict(new_state)

    def evaluate(self) -> Dict[str, float]:
        """Evaluate on test graph."""
        self.model.eval()

        with torch.no_grad():
            out = self.model(self.test_graph.x, self.test_graph.edge_index)
            pred = out.argmax(dim=1)
            correct = (pred == self.test_graph.y).sum().item()

        return {"accuracy": correct / self.test_graph.num_nodes}

    def train(self) -> List[Dict]:
        """Run federated GNN training."""
        logger.info(f"Starting federated GNN with {len(self.clients)} clients")

        for round_num in range(self.config.num_rounds):
            # Select clients
            n = min(self.config.clients_per_round, len(self.clients))
            indices = np.random.choice(len(self.clients), n, replace=False)
            selected = [self.clients[i] for i in indices]

            # Collect updates
            updates = [c.train(self.model) for c in selected]

            # Aggregate
            self.aggregate(updates)

            # Evaluate
            metrics = self.evaluate()

            # Client-level evaluation
            client_accs = [c.evaluate(self.model)["accuracy"] for c in self.clients]

            record = {
                "round": round_num,
                **metrics,
                "mean_client_acc": np.mean(client_accs),
                "num_clients": len(updates)
            }
            self.history.append(record)

            if (round_num + 1) % 10 == 0:
                logger.info(
                    f"Round {round_num + 1}: "
                    f"test_acc={metrics['accuracy']:.4f}, "
                    f"client_acc={record['mean_client_acc']:.4f}"
                )

        return self.history


def main():
    """Main entry point."""
    print("=" * 60)
    print("Tutorial 112: FL with GNNs")
    print("=" * 60)

    config = GNNConfig()
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    # Create clients with local subgraphs
    clients = []
    for i in range(config.num_clients):
        graph = GraphData.random_graph(
            num_nodes=config.nodes_per_client,
            num_features=config.node_features,
            num_classes=config.num_classes,
            edge_density=config.edge_density,
            seed=config.seed + i
        )
        client = GNNClient(i, graph, config)
        clients.append(client)

    # Test graph
    test_graph = GraphData.random_graph(
        num_nodes=200,
        num_features=config.node_features,
        num_classes=config.num_classes,
        edge_density=config.edge_density,
        seed=999
    )

    # Model
    model = FedGNN(config)
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Train
    server = GNNServer(model, clients, test_graph, config)
    history = server.train()

    print("\n" + "=" * 60)
    print("Training Complete")
    print(f"Final Test Accuracy: {history[-1]['accuracy']:.4f}")
    print(f"Final Client Accuracy: {history[-1]['mean_client_acc']:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
```

---

## Key Insights

### Federated GNN Challenges

1. **Graph partitioning**: Subgraphs may differ
2. **Message passing**: Limited by privacy
3. **Heterogeneity**: Different graph structures
4. **Scalability**: Large graphs

### Best Practices

- Use subgraph-level FL when possible
- Consider graph structure in aggregation
- Balance local and global knowledge
- Handle missing cross-client edges

---

## Exercises

1. **Exercise 1**: Implement graph-level FL
2. **Exercise 2**: Add cross-client edges with privacy
3. **Exercise 3**: Design GNN personalization
4. **Exercise 4**: Implement graph attention networks

---

## References

1. He, C., et al. (2021). FedGraphNN: A FL system for graph learning. In *MLSys Workshop*.
2. Wu, C., et al. (2021). Federated graph machine learning. *IEEE TNNLS*.
3. Zhang, K., et al. (2021). Subgraph FL: A new paradigm. In *KDD*.

---

*Copyright 2026 Olaf Yunus Laitinen Imanov and Contributors. Released under EUPL 1.2.*
