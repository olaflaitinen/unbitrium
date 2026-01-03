# Tutorial 022: Tuning FedProx Mu

## Overview
**FedProx** adds a regularization term $\frac{\mu}{2}\|w - w^t\|^2$. The parameter $\mu$ controls the trade-off.

## Sweep
- $\mu = 0$ (Equivalent to FedAvg)
- $\mu = 0.001$
- $\mu = 0.01$
- $\mu = 1.0$ (Too rigid?)

## Code

```python
import unbitrium as ub

mus = [0.0, 0.001, 0.01, 0.1, 1.0]

for mu in mus:
    agg = ub.aggregators.FedProx(mu=mu)
    # run sim...
```

## Analysis
- Low $\mu$: Similar to FedAvg, drift occurs.
- Optimal $\mu$: Constrains drift without preventing learning.
- High $\mu$: Prevents the model from moving; stagnation.
