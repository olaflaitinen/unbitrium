# Tutorial 166: Dropout in FL

This tutorial covers dropout strategies for regularization.

## Variation

- **Standard Dropout**: During client training.
- **MC Dropout**: For uncertainty estimation.

## Configuration

```yaml
model:
  dropout_rate: 0.3
```

## Exercises

1. Dropout interaction with small local datasets.
2. Federated uncertainty quantification.
