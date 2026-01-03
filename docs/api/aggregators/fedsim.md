# FedSim

Similarity-guided weighting using cosine similarity.

## Formal Definition

$$
w^{t+1} = \sum_k \mathrm{sim}(w_k^t, w_g^t)\, w_k^t
$$

## API

```python
from unbitrium.aggregators import FedSim

aggregator = FedSim(temperature=1.0)
global_update = aggregator.aggregate(client_updates, global_model)
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `temperature` | float | 1.0 | Softmax temperature for similarity weights |

## Complexity

- Time: $O(K^2 \cdot d)$ for pairwise similarities
- Space: $O(d)$

## References

- FedSim: Similarity-Guided Federated Learning.
