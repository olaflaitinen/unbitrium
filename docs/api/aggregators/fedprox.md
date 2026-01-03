# FedProx

Proximal regularization to mitigate client drift.

## Formal Definition

$$
\min_w F_k(w) + \frac{\mu}{2}\|w - w_g\|^2
$$

## API

```python
from unbitrium.aggregators import FedProx

aggregator = FedProx(mu=0.01)
global_update = aggregator.aggregate(client_updates, client_weights)
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `mu` | float | 0.01 | Proximal regularization strength |

## Complexity

- Time: $O(K \cdot d)$
- Space: $O(d)$

## Security Considerations

Same as FedAvg.

## References

- Li et al., "Federated Optimization in Heterogeneous Networks", MLSys 2020.
