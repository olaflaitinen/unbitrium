# Tutorial 047: Vertical Federated Learning (VFL) - Simulation

## Overview
VFL involves clients sharing *features* rather than *samples* (e.g., Bank has financial data, Hospital has health data for *same* users).

## Unbitrium Support
While primarily HFL, we can simulate VFL by splitting the *features* (columns) of a dataset instead of rows.

## Code
```python
# Feature Split
partitioner = ub.partitioning.VerticalFeatureSplit(num_clients=2)
```

## Note
Requires different aggregation logic (Exchange of embeddings/gradients on cut layer).
