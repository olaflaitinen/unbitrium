# Tutorial 103: Federated Continual Learning

This tutorial covers FL under concept drift (non-stationary data).

## Challenge

Client data distributions shift over time, causing catastrophic forgetting.

## Mitigation

- Elastic Weight Consolidation (EWC).
- Replay buffers.

## Configuration

```yaml
continual:
  method: "ewc"
  fisher_importance: 1000
```

## Exercises

1. Measure forgetting across federated rounds.
2. How does EWC interact with FedProx?
