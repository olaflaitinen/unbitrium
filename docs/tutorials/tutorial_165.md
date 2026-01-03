# Tutorial 165: Label Smoothing in FL

This tutorial covers label smoothing for regularization.

## Technique

Soften one-hot labels.

$$
y'_i = (1-\epsilon)y_i + \frac{\epsilon}{K}
$$

## Configuration

```yaml
training:
  label_smoothing: 0.1
```

## Exercises

1. Impact on calibration under non-IID.
2. Combination with knowledge distillation.
