# Tutorial 153: Median Aggregation

This tutorial covers coordinate-wise median for Byzantine robustness.

## Algorithm

$$
w^{t+1}_i = \text{median}(\{w^t_{k,i}\}_k)
$$

## Configuration

```yaml
aggregator:
  type: "median"
```

## Exercises

1. Breakdown point of median.
2. Computational cost vs trimmed mean.
