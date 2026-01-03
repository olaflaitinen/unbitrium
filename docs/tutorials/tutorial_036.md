# Tutorial 036: Differential Privacy (Central)

## Overview
Adding noise to the global aggregate to protect client privacy.

## Params
- $\epsilon$ (Epsilon): Privacy budget (e.g., 1.0).
- $\delta$ (Delta): Failure probability ($10^{-5}$).

## Clipping
Crucial step: Clip updates to max norm $C$ before averaging.

## Code

```python
dp = ub.privacy.DifferentialPrivacy(epsilon=1.0, delta=1e-5, max_grad_norm=1.0)
# Inside aggregator
updates = [dp.clip_gradients(u) for u in updates]
aggregate = average(updates)
aggregate = dp.add_noise(aggregate)
```

## Trade-off
Privacy comes at the cost of utility (Accuracy).
