# Tutorial 177: Effective Number of Classes

This tutorial covers measuring label diversity.

## Definition

$$
\text{ENS} = \exp\left(-\sum_i p_i \log p_i\right) = \exp(H)
$$

## Implementation

```python
def effective_classes(distribution):
    entropy = -np.sum(distribution * np.log(distribution + 1e-10))
    return np.exp(entropy)
```

## Exercises

1. Low ENS indicates class imbalance.
2. Aggregating ENS across clients.
