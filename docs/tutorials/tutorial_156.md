# Tutorial 156: FedNova Aggregation

This tutorial covers FedNova for normalized averaging.

## Algorithm

Normalizes updates by local steps to handle heterogeneous local computation.

$$
\Delta_k^{(t)} = \tau_k \cdot d_k^{(t)}
$$

## Configuration

```yaml
aggregator:
  type: "fednova"
```

## Exercises

1. Impact on stragglers with different local steps.
2. Comparison with FedAvg under heterogeneous compute.
