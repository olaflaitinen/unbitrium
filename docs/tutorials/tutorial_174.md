# Tutorial 174: Total Variation Metric

This tutorial covers total variation as heterogeneity measure.

## Definition

$$
TV(p, q) = \frac{1}{2}\sum_i |p_i - q_i|
$$

## Implementation

```python
def total_variation(p, q):
    return 0.5 * np.abs(p - q).sum()
```

## Exercises

1. Relationship to L1 distance.
2. Interpretation for multi-class distributions.
