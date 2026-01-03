# Tutorial 179: KL Divergence

This tutorial covers KL divergence for distribution comparison.

## Definition

$$
D_{KL}(P \| Q) = \sum_i P(i) \log \frac{P(i)}{Q(i)}
$$

## Implementation

```python
def kl_divergence(p, q):
    return np.sum(p * np.log((p + 1e-10) / (q + 1e-10)))
```

## Exercises

1. Asymmetry of KL divergence.
2. Symmetrized versions (JS).
