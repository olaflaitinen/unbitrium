# Tutorial 112: Federated Self-Supervised Learning

This tutorial covers FL using self-supervised pretraining.

## Approach

- SimCLR, BYOL, or MoCo-style contrastive objectives.
- No labels required on clients.

## Configuration

```yaml
ssl:
  method: "simclr"
  temperature: 0.5
  projection_dim: 128
```

## Exercises

1. Cross-client negative sampling strategies.
2. Communication efficiency with large batch contrastive.
