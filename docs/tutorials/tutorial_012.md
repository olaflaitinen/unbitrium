# Tutorial 012: Jensen-Shannon Divergence Map

## Overview
JSD is a symmetric and smoothed version of KL divergence. It is bounded $[0, 1]$ (or $[0, \ln 2]$), making it easier to interpret than KL.

## Visualization
We will create a heatmap of pairwise JSD between all clients.

## Code

```python
import unbitrium as ub
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Partition
client_data = ub.partitioning.DirichletLabelSkew(alpha=0.1, num_clients=10).partition(ub.datasets.load("cifar10"))

# Extract distributions
dists = []
targets = ub.partitioning.base.Partitioner(1)._get_targets(ub.datasets.load("cifar10")[0])
for indices in client_data.values():
    c = np.bincount(targets[indices], minlength=10)
    dists.append(c / c.sum())

# Pairwise Matrix
matrix = np.zeros((10, 10))
for i in range(10):
    for j in range(10):
        matrix[i,j] = ub.metrics.compute_js_divergence(dists[i], dists[j])

# Plot
sns.heatmap(matrix, annot=True)
plt.title("Pairwise JS Divergence")
plt.show()
```

## Insight
Blocks on the diagonal (if sorted by dominant class) indicate clusters of similar clients. This can inform **Clustered Federated Learning**.
