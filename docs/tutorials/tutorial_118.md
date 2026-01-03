# Tutorial 118: Federated Clustering

This tutorial covers FL for unsupervised clustering.

## Approach

- Federated K-Means: Local centroids aggregated globally.
- Privacy: Only centroids exchanged.

## Configuration

```yaml
clustering:
  method: "kmeans"
  k: 10
  local_iterations: 5
```

## Exercises

1. Handling different local optima.
2. Privacy of cluster assignments.
