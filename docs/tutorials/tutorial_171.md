# Tutorial 171: Early Stopping in FL

This tutorial covers early stopping strategies.

## Challenge

Global validation may not reflect local performance.

## Strategies

- Stop on global validation plateau.
- Patience-based stopping.
- Per-client early stopping.

## Configuration

```yaml
early_stopping:
  patience: 10
  min_delta: 0.001
```

## Exercises

1. Overfitting detection in FL.
2. Aggregating stopping decisions.
