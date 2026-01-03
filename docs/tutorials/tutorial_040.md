# Tutorial 040: Client Selection Bias and Fairness

## Overview
Random selection isn't fair if some clients are often offline (Systems Heterogeneity).

## Fairness Metric
Jain's Fairness Index on "number of times selected".

## Experiment
Compare `RandomScheduler` vs `PowerOfChoice`.

## Code

```python
selection_counts = np.zeros(num_clients)
# Run sim
# ...
jains_index = (np.sum(counts)**2) / (len(counts) * np.sum(counts**2))
```

## Goal
Maximize Jain's Index while maintaining convergence speed.
