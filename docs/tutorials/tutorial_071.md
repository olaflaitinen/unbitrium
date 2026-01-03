# Tutorial 071: Debugging NaN Gradients

## Overview
Gradients exploding to NaN/Inf is common in FL due to unstable local updates.

## Detection
Unbitrium has a `SafeAggregator` wrapper.

## Code
```python
agg = ub.aggregators.SafeAggregator(
    ub.aggregators.FedAvg(),
    check_nan=True,
    check_inf=True
)
```

## Behavior
If a client sends NaNs, drop that update and log a warning.
