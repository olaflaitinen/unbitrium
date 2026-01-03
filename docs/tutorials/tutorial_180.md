# Tutorial 180: Hellinger Distance

This tutorial covers Hellinger distance for distribution comparison.

## Definition

$$
H(P, Q) = \frac{1}{\sqrt{2}} \sqrt{\sum_i (\sqrt{p_i} - \sqrt{q_i})^2}
$$

## Implementation

```python
def hellinger(p, q):
    return np.sqrt(0.5 * np.sum((np.sqrt(p) - np.sqrt(q))**2))
```

## Exercises

1. Relationship to other f-divergences.
2. Advantages for numerical stability.
