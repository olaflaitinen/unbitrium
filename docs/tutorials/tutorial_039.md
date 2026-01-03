# Tutorial 039: Privacy Budget Accounting

## Overview
Tracking the cumulative $\epsilon$ consumed over $T$ rounds.

## Composition
Advanced composition theorems (RDP - Rényi DP) provide tighter bounds than basic sequential composition.

## Code
```python
# Unbitrium creates a "Privacy Accountant"
accountant.step()
current_epsilon = accountant.get_privacy_spent(delta=1e-5)
```

## Stopping condition
Stop training when $\epsilon > 10.0$.
