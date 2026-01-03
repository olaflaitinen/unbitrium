# Tutorial 168: Gradient Clipping in FL

This tutorial covers gradient clipping strategies.

## Purpose

- Prevent exploding gradients.
- Required for Differential Privacy.

## Configuration

```yaml
training:
  gradient_clip_norm: 1.0
```

## Exercises

1. Impact on convergence rate.
2. Interaction with adaptive optimizers.
