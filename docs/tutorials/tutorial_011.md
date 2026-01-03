# Tutorial 011: Earth Mover's Distance (EMD) Analysis

## Overview
EMD is a powerful metric for quantifying the distance between the local label distribution of a client and the global prior.

## Setup
- **Dataset**: MNIST
- **Partitioner**: Dirichlet ($\alpha=0.5$)
- **Goal**: Correlation between EMD and local test accuracy.

## Code

```python
import unbitrium as ub
import numpy as np

dataset = ub.datasets.load("mnist")
# Global prior (uniform for balanced MNIST)
global_dist = np.ones(10) / 10.0

partitioner = ub.partitioning.DirichletLabelSkew(alpha=0.5, num_clients=20)
client_data = partitioner.partition(dataset)

emds = []
for cid, indices in client_data.items():
    # Compute local distribution
    labels = partitioner._get_targets(dataset)[indices]
    counts = np.bincount(labels, minlength=10)
    local_dist = counts / counts.sum()

    # Compute EMD
    val = ub.metrics.compute_emd(local_dist, global_dist)
    emds.append(val)
    print(f"Client {cid}: EMD={val:.4f}")

print(f"Mean EMD: {np.mean(emds)}")
```

## Interpretation
Higher EMD $\implies$ Data is less representative of the global problem. We expect clients with high EMD to have higher local loss on the global test set.
