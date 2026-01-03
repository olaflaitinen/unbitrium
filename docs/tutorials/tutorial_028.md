# Tutorial 028: Robust Aggregation with Krum

## Overview
What if some clients are Byzantine (malicious or broken)? **Krum** selects one update that minimizes the sum of squared distances to neighbors.

## Attack Simulation
Inject random noise or "label flip" updates from 20% of clients.

## Code

```python
# Simulate attack in client training phase (conceptually)
# ...

# Use Krum
agg = ub.aggregators.Krum(num_byzantine=2) # Assume at most 2 attackers
```

## Result
FedAvg fails (0% accuracy on attacked classes). Krum maintains performance, although it may be slower to converge (selects only 1 update).
