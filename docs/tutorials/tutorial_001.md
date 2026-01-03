# Tutorial 001: Introduction to Dirichlet Label Skew Partitioning

## Overview
This tutorial demonstrates the fundamental non-IID partitioning strategy: **Dirichlet Label Skew**. We will simulate a federated learning environment where client data distributions are sampled from a Dirichlet distribution.

## Experimental Setup
- **Dataset**: CIFAR-10
- **Partitioner**: `DirichletLabelSkew`
- **Aggregator**: `FedAvg`
- **Heterogeneity Metric**: Earth Mover's Distance (EMD)

## Parameters
- `alpha`: 0.1 (High heterogeneity)
- `num_clients`: 10

## Code Implementation

```python
import unbitrium as ub
import numpy as np

# 1. Load Dataset
dataset = ub.datasets.load("cifar10")

# 2. Partition Data
partitioner = ub.partitioning.DirichletLabelSkew(alpha=0.1, num_clients=10, seed=42)
client_datasets = partitioner.partition(dataset)

# 3. Analyze Heterogeneity
metrics = ub.metrics.compute_all(client_datasets)
print(f"Average EMD: {metrics['emd_avg']:.4f}")

# 4. Run Simulation
config = ub.core.SimulationConfig(num_rounds=10, clients_per_round=5)
engine = ub.core.SimulationEngine(config, ub.aggregators.FedAvg())
results = engine.run(client_datasets)
```

## Expected Results
With $\alpha=0.1$, we expect high EMD values (>1.5) indicating strong non-IIDness. Convergence will be slower compared to IID settings.
