# Tutorial 100: Defending against Gradient Leakage

This tutorial demonstrates defenses like Gradient Pruning.

## Defense

$$
\nabla W_{pruned} = \nabla W \odot M, \quad M_{ij} = \mathbb{I}(|\nabla W_{ij}| > \tau)
$$

## Implementation

```python
def prune_gradient(grad, percentage=0.1):
    k = int(grad.numel() * percentage)
    threshold = torch.topk(grad.abs().flatten(), k, largest=False).values.max()
    grad[grad.abs() < threshold] = 0
    return grad
```

## Exercises

1. Evaluate DLG success rate against pruned gradients.
2. Trade-off between pruning percentage and model accuracy.
