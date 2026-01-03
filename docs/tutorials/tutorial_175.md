# Tutorial 175: Centered Kernel Alignment

This tutorial covers CKA for comparing representations.

## Definition

$$
CKA(K, L) = \frac{HSIC(K, L)}{\sqrt{HSIC(K,K) \cdot HSIC(L,L)}}
$$

## Use Case

Compare learned representations across clients.

## Implementation

```python
def cka(X, Y):
    # Linear CKA
    XX = X @ X.T
    YY = Y @ Y.T
    XY = X @ Y.T
    return np.trace(XX @ YY) / (np.linalg.norm(XX, 'fro') * np.linalg.norm(YY, 'fro'))
```

## Exercises

1. CKA for neural network layer comparison.
2. Tracking representation drift.
