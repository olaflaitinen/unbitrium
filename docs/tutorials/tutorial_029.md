# Tutorial 029: Trimmed Mean vs Outliers

## Overview
**Trimmed Mean** removes the largest and smallest $k$ values for each parameter coordinate before averaging.

## Scenario
Some clients have hardware faults sending $10^9$ values.

## Code

```python
agg = ub.aggregators.TrimmedMean(beta=0.1) # Trim 10% from top and bottom
```

## Result
Extremely effective against "amplitude" attacks or bugs. Less effective against "direction" attacks (model poisoning) compared to Krum.
