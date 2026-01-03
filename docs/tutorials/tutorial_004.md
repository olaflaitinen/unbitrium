# Tutorial 004: Mixture-of-Dirichlet-Multinomials (MoDM)

## Overview
Standard Dirichlet skew assumes a single global prior. MoDM simulates distinct subgroups of clients (e.g., urban vs rural) by drawing from a mixture of Dirichlet priors.

## Experimental Setup
- **Dataset**: CIFAR-100 (Coarse labels)
- **Partitioner**: `MixtureOfDirichletMultinomials`
- **Mixtures**: 3 components

## Implementation

```python
import unbitrium as ub

partitioner = ub.partitioning.MixtureOfDirichletMultinomials(
    num_clients=30,
    num_mixtures=3,
    alpha=0.5
)
# Note: Using CIFAR-100 for richer class structure
client_datasets = partitioner.partition(ub.datasets.load("cifar100"))

# Compute Heterogeneity
metrics = ub.metrics.compute_all(client_datasets)
print(f"JSD between communities: {metrics.get('inter_community_jsd')}")
```

## Discussion
MoDM creates clusters of clients with similar distributions. This is ideal for testing personalized FL algorithms like clustered federated learning (CFL).
