# Tutorial 130: Federated Unlearning

This tutorial covers removing data influence from a trained federated model.

## Challenge

- A client requests data deletion.
- Model must "forget" their contribution.

## Approaches

- **Retraining from scratch**: Expensive.
- **Influence Removal**: Approximate via influence functions.
- **SISA**: Sharded training for efficient unlearning.

## Configuration

```yaml
unlearning:
  method: "sisa"
  shards: 10
```

## Exercises

1. Verification that unlearning succeeded.
2. Trade-off between unlearning cost and accuracy.
