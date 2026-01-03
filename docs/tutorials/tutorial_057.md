# Tutorial 057: Agnostic Federated Learning (AFL)

## Overview
Optimize for the distribution of the *unknown* target distribution, modeled as a mixture of client distributions.

## Minimax
Minimax optimization: $\min_w \max_{\lambda} \sum \lambda_k F_k(w)$.

## Interpretation
Protects against distribution shift at test time by optimizing for the worst-case mixture of training clients.

## Code
```python
agg = ub.aggregators.AgnosticFL()
```
