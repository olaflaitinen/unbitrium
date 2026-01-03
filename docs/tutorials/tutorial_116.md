# Tutorial 116: Federated Anomaly Detection

This tutorial covers FL for anomaly/outlier detection.

## Approach

- Autoencoder-based reconstruction error.
- One-class classification (SVDD).

## Configuration

```yaml
model:
  type: "autoencoder"
  latent_dim: 16

anomaly:
  threshold_percentile: 95
```

## Exercises

1. Challenge of rare anomalies across non-IID clients.
2. Federated threshold calibration.
