# Tutorial 002: Impact of Alpha on Dirichlet Partitioning

## Overview
This tutorial explores how the concentration parameter $\alpha$ in Dirichlet partitioning affects data heterogeneity and model convergence.

## Experimental Question
How does varying $\alpha \in \{0.1, 1.0, 10.0\}$ impact the convergence rate of FedAvg on CIFAR-10?

## Configuration
- **Dataset**: CIFAR-10
- **Partitioner**: `DirichletLabelSkew`
- **Metric**: accuracy vs round

## Implementation

```python
import unbitrium as ub

alphas = [0.1, 1.0, 10.0]
results = {}

for alpha in alphas:
    partitioner = ub.partitioning.DirichletLabelSkew(alpha=alpha, num_clients=10)
    client_datasets = partitioner.partition(ub.datasets.load("cifar10"))

    engine = ub.core.SimulationEngine(
        ub.core.SimulationConfig(num_rounds=20),
        ub.aggregators.FedAvg()
    )
    results[alpha] = engine.run(client_datasets)
```

## Analysis
- **Low Alpha (0.1)**: High heterogeneity, clients have 1-2 classes dominantly. Convergence is unstable.
- **High Alpha (10.0)**: Near-IID. Convergence is smooth and fast.
