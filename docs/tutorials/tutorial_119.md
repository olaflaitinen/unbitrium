# Tutorial 119: Federated Principal Component Analysis

This tutorial covers FL for dimensionality reduction.

## Approach

- Distributed SVD or power iteration.
- Aggregate local covariance matrices.

## Configuration

```yaml
pca:
  n_components: 10
  method: "distributed_svd"
```

## Exercises

1. Communication cost of covariance matrices.
2. Privacy implications of sharing covariances.
