# Tutorial 018: Imbalance Ratio Impact

## Overview
Defined as $\rho = \frac{\max_c n_c}{\min_c n_c}$ for a client's local data.

## Question
At what Imbalance Ratio does local training fail to produce useful features?

## Experiment
Sweep $\rho$ from 1 (balanced) to 100 on Client 1, keeping others fixed. Measure Client 1's validation loss.

## Code (Concept)
```python
# Create custom datasets with specific imbalance ratios
# ...
# ub.metrics.compute_imbalance_ratio(dataset) -> returns float
```

## Results
High imbalance often leads to the client ignoring minority classes, submitting updates that "forget" those classes in the global model (Catastrophic Forgetting).
