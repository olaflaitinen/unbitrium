# Tutorial 008: Temporal Shift Concept Drift

## Overview
Simulating "Concept Drift" where data distribution changes over *time* (Rounds).

## Strategy
We simulate drift by rotating the partition seeds or re-partitioning every $N$ rounds.

## Code

```python
import unbitrium as ub

dataset = ub.datasets.load("mnist")
config = ub.core.SimulationConfig(num_rounds=20)
state = ub.aggregators.FedAvg()

# Initial Partition
partitioner_A = ub.partitioning.DirichletLabelSkew(alpha=0.1, seed=1)
partitioner_B = ub.partitioning.DirichletLabelSkew(alpha=0.1, seed=2) # Different distribution

data_A = partitioner_A.partition(dataset)
data_B = partitioner_B.partition(dataset)

model = None

for r in range(20):
    # Drift at round 10
    current_data = data_A if r < 10 else data_B

    # Manual stepped execution
    engine = ub.core.SimulationEngine(
        ub.core.SimulationConfig(num_rounds=1), # Single round
        state,
        model=model,
        client_datasets=current_data
    )
    res = engine.run()
    model = engine.global_model

    if r == 10:
        print("!!! CONCEPT DRIFT OCCURRED !!!")
```

## Observation
Accuracy usually drops sharply at Round 10 as the global model is no longer aligned with the new local distributions ($data_B$).
