# Tutorial 113: Federated Active Learning

This tutorial covers querying strategies for FL.

## Approach

- Clients propose samples for labeling.
- Server selects most informative across all clients.

## Configuration

```yaml
active_learning:
  strategy: "uncertainty"
  budget_per_round: 100
```

## Exercises

1. Privacy implications of revealing uncertainty scores.
2. Balancing exploration vs exploitation in sample selection.
