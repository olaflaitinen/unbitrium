# Tutorial 009: Extreme Partitioning (One Class Per Client)

## Overview
The "Pathological" case often used in early FL papers (e.g., McMahan 2017). Each client holds samples from only 1 or 2 classes.

## Setup
- **Dataset**: MNIST
- **Splitting**: Sort by label, divide into shards, assign 2 shards per client.

## Unbitrium Implementation
We can use `DirichletLabelSkew` with extremely low $\alpha$.

```python
import unbitrium as ub

# Alpha -> 0 implies almost 1 class per client
partitioner = ub.partitioning.DirichletLabelSkew(alpha=0.0001, num_clients=100)
client_datasets = partitioner.partition(ub.datasets.load("mnist"))

# Metric Check
metrics = ub.metrics.compute_all(client_datasets)
print(f"Avg Effective Classes: {metrics['avg_effective_classes']}") # Should be ~1.0
```

## Challenge
Gradients from clients point in orthogonal directions. Aggregation (averaging) can be destructive. `FedAvg` with standard params often fails to converge; `FedProx` or lower learning rates are required.
