# Tutorial 023: FedDyn - No Hyperparameters?

## Overview
**FedDyn** (Acar et al., 2021) claims to be robust to parameter choices compared to FedProx. It maintains alignment via dynamic regularization variables.

## Comparison
Run FedProx (tuned) vs FedDyn (default $\alpha=0.1$).

## Code

```python
import unbitrium as ub

agg_dyn = ub.aggregators.FedDyn(alpha=0.1)
# run sim
```

## Result
FedDyn often achieves faster convergence in terms of communication rounds, at the cost of slight compute overhead (storing state).
