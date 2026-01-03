# Tutorial 105: Federated Recommendation Systems

This tutorial explores FL for collaborative filtering.

## Setting

- Clients hold user interaction histories.
- Server aggregates without accessing raw interactions.

## Model

Matrix Factorization or Neural Collaborative Filtering.

## Configuration

```yaml
model:
  type: "ncf"
  embedding_dim: 64

partitioning:
  user_level: true
```

## Exercises

1. Privacy vs utility trade-off in recommendation FL.
2. Cold-start problem handling.
