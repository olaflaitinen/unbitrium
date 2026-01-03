# Tutorial 152: FedAdagrad Aggregator

This tutorial covers the FedAdagrad server optimizer.

## Algorithm

$$
v_t = v_{t-1} + g_t^2
$$
$$
w_{t+1} = w_t - \frac{\eta}{\sqrt{v_t} + \tau} g_t
$$

## Configuration

```yaml
aggregator:
  type: "fedadagrad"
  learning_rate: 0.01
  tau: 1e-3
```

## Exercises

1. Accumulating second moments over rounds.
2. Comparison with client-side Adagrad.
