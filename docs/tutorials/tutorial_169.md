# Tutorial 169: Per-Sample Gradient Computation

This tutorial covers computing gradients per sample for DP.

## Requirement

DP-SGD requires per-sample gradients for clipping.

## Implementation

```python
# Using functorch
from functorch import vmap, grad

per_sample_grads = vmap(grad(loss_fn), in_dims=(None, 0))(params, batch)
```

## Exercises

1. Memory overhead of per-sample gradients.
2. Efficient implementations.
