# FedAvg

Classical weighted average aggregator baseline.

## Formal Definition

$$
w^{t+1} = \sum_k \frac{n_k}{\sum_j n_j} w_k^t
$$

## API

```python
from unbitrium.aggregators import FedAvg

aggregator = FedAvg()
global_update = aggregator.aggregate(client_updates, client_weights)
```

## Parameters

None required.

## Complexity

- Time: $O(K \cdot d)$ where $K$ is number of clients and $d$ is model dimension.
- Space: $O(d)$

## Security Considerations

FedAvg requires access to client model updates. Consider Secure Aggregation for privacy.

## References

- McMahan et al., "Communication-Efficient Learning of Deep Networks from Decentralized Data", AISTATS 2017.
