# Tutorial 173: Cosine Annealing in FL

This tutorial covers cosine LR schedules in FL.

## Schedule

$$
\eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})(1 + \cos(\frac{t}{T}\pi))
$$

## Configuration

```yaml
training:
  lr_schedule: "cosine"
  T_max: 100
  eta_min: 1e-6
```

## Exercises

1. Warm restarts in FL.
2. Coordinate server and client schedules.
