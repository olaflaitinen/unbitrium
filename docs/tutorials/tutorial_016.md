# Tutorial 016: Weight Drift Visualization

## Overview
Visualizing how far local models drift from the global model during local epochs.

## Metric
Euclidean Norm: $\| w_k^t - w_{global}^t \|_2$.

## Experiment
Compare 1 local epoch vs 10 local epochs.

## Code

```python
import unbitrium as ub

drift_norms = []

# Mock loop
w_global = np.zeros(100)
w_local_1epoch = np.random.normal(0, 0.1, 100)
w_local_10epoch = np.random.normal(0, 1.0, 100) # Further away

d1 = ub.metrics.compute_drift_norm(w_local_1epoch, w_global)
d10 = ub.metrics.compute_drift_norm(w_local_10epoch, w_global)

print(f"Drift (1 epoch): {d1}, Drift (10 epochs): {d10}")
```

## Significance
This justifies algorithms like **FedProx** or **SCAFFOLD** which add terms to constrain this specific drift.
