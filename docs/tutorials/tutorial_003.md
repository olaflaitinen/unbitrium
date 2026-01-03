# Tutorial 003: Quantity Skew with Power Law Distribution

## Overview
Real-world FL clients often have vastly different amounts of data. This tutorial demonstrates the `QuantitySkewPowerLaw` partitioner.

## Setup
We simulate a scenario where dataset sizes follow a power law $P(x) \propto x^{-\gamma}$.

## Parameters
- `gamma`: 0.5
- `num_clients`: 50
- `aggregator`: `FedAvg` (weighted by sample size)

## Code

```python
import unbitrium as ub

# Partition with Quantity Skew
partitioner = ub.partitioning.QuantitySkewPowerLaw(gamma=0.5, num_clients=50)
client_datasets = partitioner.partition(ub.datasets.load("mnist"))

# Check sizes
sizes = [len(idx) for idx in client_datasets.values()]
print(f"Max size: {max(sizes)}, Min size: {min(sizes)}")

# Run
engine = ub.core.SimulationEngine(
    ub.core.SimulationConfig(num_rounds=15),
    ub.aggregators.FedAvg()
)
engine.run(client_datasets)
```

## Insights
The "whale" clients (those with most data) dominate the global model update in FedAvg. Small clients may be effectively ignored.
