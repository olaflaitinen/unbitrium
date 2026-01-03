# Tutorial 189: Synthetic Data Generation

This tutorial covers generating synthetic federated datasets.

## Approaches

- Controlled label skew.
- Feature drift simulation.
- Time-varying distributions.

## Code

```python
from unbitrium.partitioning import DirichletLabelSkew

partitioner = DirichletLabelSkew(
    alpha=0.5,
    num_clients=100,
    seed=42
)
client_data = partitioner.partition(base_dataset)
```

## Exercises

1. Designing experiments with controlled heterogeneity.
2. Ablation studies on alpha.
