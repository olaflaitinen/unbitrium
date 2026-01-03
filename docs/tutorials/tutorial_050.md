# Tutorial 050: Contributing a New Aggregator

## Overview
How to add `YourNewAlgo` to Unbitrium.

## Steps
1. Inherit from `ub.aggregators.Aggregator`.
2. Implement `aggregate(updates, global_model)`.
3. Register using decorator (if applicable) or add to `__init__.py`.
4. Add Test in `tests/test_aggregators.py`.
5. Add Docstring.

## Checklist
- Does it handle empty updates?
- Does it respect `max_grad_norm`?
- Vectorized?
