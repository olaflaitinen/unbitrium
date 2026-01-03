# Tutorial 157: SCAFFOLD

This tutorial covers SCAFFOLD for variance reduction.

## Algorithm

Uses control variates to correct local updates.

$$
\Delta_k = g_k - c_k + c
$$

## Configuration

```yaml
aggregator:
  type: "scaffold"
```

## Exercises

1. Communication overhead of control variates.
2. Convergence improvement vs FedAvg.
