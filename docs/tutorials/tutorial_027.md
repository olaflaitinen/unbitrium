# Tutorial 027: FedAdam (Server-Side Optimization)

## Overview
**FedAdam** treats client updates as a "gradient" for the server's own Adam optimizer.

## Configuration
- Client LR: 0.1
- Server LR: 0.01 (FedAdam step size)

## Implementation

```python
agg = ub.aggregators.FedAdam(learning_rate=0.01, beta1=0.9, beta2=0.99)
```

## Comparisons
FedAdam converges significantly faster than FedAvg on Transformer language models (e.g., Federated NLP).
