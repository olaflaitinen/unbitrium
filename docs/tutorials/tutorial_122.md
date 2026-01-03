# Tutorial 122: Client Selection Strategies

This tutorial covers advanced client selection policies.

## Strategies

- **Random**: Uniform sampling.
- **Power-of-Choice**: Sample extra, select best.
- **Importance Sampling**: Based on loss or data size.

## Configuration

```yaml
selection:
  strategy: "power_of_choice"
  oversample_factor: 2
```

## Exercises

1. Bias introduced by non-uniform selection.
2. Impact on convergence speed.
