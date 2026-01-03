# Tutorial 172: Learning Rate Warmup in FL

This tutorial covers warmup strategies for FL.

## Purpose

Stabilize early training, especially with large batches.

## Configuration

```yaml
training:
  warmup_rounds: 5
  warmup_factor: 0.1
```

## Exercises

1. Warmup for server vs client LR.
2. Interaction with heterogeneity.
