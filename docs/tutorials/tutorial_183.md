# Tutorial 183: CIFAR Partitioning

This tutorial covers CIFAR-10/100 partitioning for FL.

## Standard Setups

- Dirichlet $\alpha \in \{0.1, 0.5, 1.0\}$.
- 2-class per client (pathological).
- IID baseline.

## Configuration

```yaml
dataset:
  name: "cifar10"

partitioning:
  strategy: "dirichlet"
  alpha: 0.5
  num_clients: 100
```

## Exercises

1. Visualize class distributions.
2. Compare $\alpha$ values on convergence.
