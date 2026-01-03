# Tutorial 021: FedAvg Baseline Baseline

## Overview
Establishing the baseline: **Federated Averaging** (McMahan et al., 2017).

## Scenario
- **Task**: IID vs Non-IID CIFAR-10.
- **Aggregator**: `FedAvg`.
- **Goal**: Observe the "non-IID performance gap".

## Code

```python
import unbitrium as ub

# 1. IID Run
iid_data = ub.partitioning.DirichletLabelSkew(alpha=100.0, num_clients=10).partition(ub.datasets.load("cifar10"))
eng_iid = ub.core.SimulationEngine(ub.core.SimulationConfig(num_rounds=50), ub.aggregators.FedAvg())
res_iid = eng_iid.run(iid_data)

# 2. Non-IID Run
non_iid_data = ub.partitioning.DirichletLabelSkew(alpha=0.1, num_clients=10).partition(ub.datasets.load("cifar10"))
eng_niid = ub.core.SimulationEngine(ub.core.SimulationConfig(num_rounds=50), ub.aggregators.FedAvg())
res_niid = eng_niid.run(non_iid_data)

# Compare Accuracy Curves
```

## Takeaway
FedAvg degrades significantly (e.g., -10% accuracy) under skew.
