# Tutorial 005: Feature Shift via Clustering

## Overview
Non-IID is not just about labels ("Concept Drift" vs "Covariate Shift"). This tutorial uses `FeatureShiftClustering` to partition data based on input features, simulating covariate shift.

## Methodology
1. Extract features using a pre-trained encoder (or use raw pixel data for simple datasets).
2. Cluster features using K-Means.
3. Assign clusters to clients.

## Code

```python
import unbitrium as ub

# Feature Shift Partitioning
partitioner = ub.partitioning.FeatureShiftClustering(
    num_clients=10,
    n_clusters=10, # 1 cluster per client = extreme shift
    seed=123
)
client_datasets = partitioner.partition(ub.datasets.load("mnist"))

# Run FedProx to handle shift
config = ub.core.SimulationConfig(num_rounds=20)
aggregator = ub.aggregators.FedProx(mu=0.1)
engine = ub.core.SimulationEngine(config, aggregator)
engine.run(client_datasets)
```

## Results
Feature shift often leads to conflicting gradients. FedProx ($\mu > 0$) usually outperforms FedAvg here by restricting local deviation.
