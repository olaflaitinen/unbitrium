# Tutorial 162: Lookahead Optimization in FL

This tutorial covers Lookahead as server optimizer.

## Algorithm

Keep slow and fast weights, interpolate.

$$
\phi_{t+1} = \phi_t + \alpha(\theta_t - \phi_t)
$$

## Configuration

```yaml
server:
  optimizer: "lookahead"
  alpha: 0.5
  k: 5
```

## Exercises

1. Stability benefits.
2. Combination with FedAdam.
