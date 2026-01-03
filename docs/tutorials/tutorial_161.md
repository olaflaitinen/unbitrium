# Tutorial 161: Local Adaptivity in FL

This tutorial covers client-side adaptive optimizers.

## Approaches

- **Local Adam**: Each client uses Adam.
- **FedLAMB**: Layerwise adaptive learning rates.

## Configuration

```yaml
client:
  optimizer: "adam"
  lr: 0.001
  beta1: 0.9
  beta2: 0.999
```

## Exercises

1. Sharing optimizer states across rounds.
2. Memory overhead of client-side adaptivity.
