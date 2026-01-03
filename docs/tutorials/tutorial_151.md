# Tutorial 151: FedYogi Aggregator

This tutorial covers the FedYogi server optimizer.

## Algorithm

$$
v_t = v_{t-1} - (1-\beta_2)\text{sign}(g_t^2 - v_{t-1})g_t^2
$$

## Configuration

```yaml
aggregator:
  type: "fedyogi"
  beta1: 0.9
  beta2: 0.99
  tau: 1e-3
```

## Exercises

1. When does FedYogi outperform FedAdam?
2. Impact of tau on stability.
