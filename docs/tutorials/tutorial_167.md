# Tutorial 167: Weight Decay in FL

This tutorial covers L2 regularization in federated settings.

## Challenge

Weight decay interacts with FedProx proximal term.

## Configuration

```yaml
optimizer:
  weight_decay: 1e-4
```

## Exercises

1. Decoupled weight decay (AdamW style).
2. Optimal weight decay under heterogeneity.
