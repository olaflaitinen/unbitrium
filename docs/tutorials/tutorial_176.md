# Tutorial 176: Cosine Disagreement Metric

This tutorial covers cosine disagreement for gradient heterogeneity.

## Definition

$$
\text{Disagreement} = 1 - \frac{\sum_k \langle g_k, \bar{g} \rangle}{K \|g_k\| \|\bar{g}\|}
$$

## Implementation

```python
def cosine_disagreement(gradients, mean_grad):
    total = 0
    for g in gradients:
        cos = torch.nn.functional.cosine_similarity(
            g.flatten().unsqueeze(0),
            mean_grad.flatten().unsqueeze(0)
        )
        total += (1 - cos.item())
    return total / len(gradients)
```

## Exercises

1. Relationship to convergence issues.
2. Using disagreement for client weighting.
