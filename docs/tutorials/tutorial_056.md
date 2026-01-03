# Tutorial 056: Fairness-Aware Aggregation (q-FedAvg)

## Overview
Based on *Fair Resource Allocation in Federated Learning* (Li et al., 2020). Use an objective that penalizes high loss clients more.

## Objective
$\min_w \sum_k \frac{n_k}{n} F_k(w)^{q+1}$ (roughly).

## Effect
Forces the model to improve performance on the "worst-off" clients, improving fairness (max-min fairness).

## Code
```python
agg = ub.aggregators.qFedAvg(q=2.0)
```
