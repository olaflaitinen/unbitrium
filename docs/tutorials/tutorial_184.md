# Tutorial 184: MNIST Partitioning

This tutorial covers MNIST for FL baselines.

## Standard Setups

- Sort and partition (2 classes per client).
- Dirichlet sampling.
- IID shuffle.

## Configuration

```yaml
dataset:
  name: "mnist"

partitioning:
  strategy: "shard"
  shards_per_client: 2
```

## Exercises

1. MLP vs CNN on partitioned MNIST.
2. Synthetic MNIST variants.
