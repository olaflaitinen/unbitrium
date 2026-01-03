# Tutorial 181: Wasserstein Distance

This tutorial covers Wasserstein (EMD) for optimal transport.

## Definition

$$
W_p(P, Q) = \left(\inf_{\gamma \in \Gamma(P,Q)} \int c(x,y)^p d\gamma(x,y)\right)^{1/p}
$$

## Use in FL

Measure label distribution shift between clients and global.

## Implementation

```python
from scipy.stats import wasserstein_distance

emd = wasserstein_distance(client_labels, global_labels)
```

## Exercises

1. 1D vs multi-dimensional Wasserstein.
2. Sliced Wasserstein for efficiency.
