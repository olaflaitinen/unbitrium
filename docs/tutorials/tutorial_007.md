# Tutorial 007: Combined Label Skew and Quantity Skew

## Overview
Real-world data often exhibits *both* label skew (distribution) and quantity skew (volume). This tutorial combines two partitioners.

## Strategy
We chain partitioners (conceptually) or implement a custom logic. Unbitrium supports this by applying a Quantity Skew mask *after* a Dirichlet split, or vice-versa. Here we demonstrate a manual composition.

## Code

```python
import unbitrium as ub
import numpy as np

# 1. Get Dirichlet split
dir_partitioner = ub.partitioning.DirichletLabelSkew(alpha=0.5, num_clients=20)
dir_indices = dir_partitioner.partition(ub.datasets.load("cifar10"))

# 2. Apply Quantity Skew (Subsampling)
# We calculate desired sizes based on power law
ranks = np.arange(1, 21)
probs = ranks ** -0.5
retention_rates = probs / probs.max() # Top rank keeps 100%, others less

final_indices = {}
rng = np.random.default_rng(42)

for client_id, indices in dir_indices.items():
    rate = retention_rates[client_id]
    num_keep = int(len(indices) * rate)
    kept = rng.choice(indices, size=num_keep, replace=False)
    final_indices[client_id] = list(kept)

# Run Simulation
# ...
```

## Impact
This tests the aggregator's ability to handle highly unbalanced importance weights alongside divergent local objectives.
