# Tutorial 126: Batch Normalization in FL

This tutorial covers challenges of BatchNorm in federated settings.

## Problem

- BatchNorm statistics computed per-client diverge.
- Aggregating BN parameters hurts performance.

## Solutions

- **Group Normalization**: Batch-independent.
- **FedBN**: Keep BN local, aggregate other layers.

## Configuration

```yaml
model:
  normalization: "group_norm"
  groups: 8
```

## Exercises

1. Compare GroupNorm vs FedBN.
2. Impact on convergence under high heterogeneity.
