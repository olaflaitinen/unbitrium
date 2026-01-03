# Tutorial 178: Imbalance Ratio

This tutorial covers imbalance ratio for skew measurement.

## Definition

$$
IR = \frac{\max_k n_k}{\min_k n_k}
$$

## Implementation

```python
def imbalance_ratio(counts):
    return max(counts) / (min(counts) + 1e-10)
```

## Exercises

1. Limitations of IR for multi-class settings.
2. Alternatives: Gini index.
