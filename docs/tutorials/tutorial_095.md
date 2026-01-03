# Tutorial 095: Fairness-aware Aggregation with q-FedAvg

This tutorial implements q-FedAvg to improve fairness (reduce variance of accuracy across clients).

## Objective

$$
\min_w \sum_k \frac{n_k}{N} F_k^{q+1}(w)
$$

## Implementation

```python
def q_fedavg(updates, q=0.1, lr=0.1):
    # Reweight updates based on loss
    losses = [u['loss'] for u in updates]
    weights = [l**(q) for l in losses]
    norm = sum(weights)
    weights = [w/norm for w in weights]

    # Weighted average
    # ...
```

## Exercises

1. Relationship between $q$ and fairness.
2. Impact of $q$ on convergence.
