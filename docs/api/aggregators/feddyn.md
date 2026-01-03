# FedDyn

Dynamic regularization for improved convergence under heterogeneity.

## Formal Definition

$$
F_k(w) - \langle a_k^t, w \rangle + \frac{\alpha}{2}\|w\|^2
$$

## API

```python
from unbitrium.aggregators import FedDyn

aggregator = FedDyn(alpha=0.01)
global_update = aggregator.aggregate(client_updates, client_weights)
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `alpha` | float | 0.01 | Regularization coefficient |

## State

Maintains per-client dual variables $a_k$.

## Complexity

- Time: $O(K \cdot d)$
- Space: $O(K \cdot d)$ (for dual variables)

## References

- Acar et al., "Federated Learning Based on Dynamic Regularization", ICLR 2021.
