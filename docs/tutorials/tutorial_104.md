# Tutorial 104: Federated Graph Neural Networks

This tutorial covers FL on graph-structured data.

## Setting

- Clients hold subgraphs (e.g., social networks, molecular graphs).
- Challenge: Missing cross-client edges.

## Configuration

```yaml
model:
  type: "gcn"
  layers: 2

partitioning:
  strategy: "metis_subgraph"
```

## Exercises

1. How to handle missing edges at graph partitions?
2. FedAvg adaptation for GNN weight averaging.
