# Tutorial 193: MLP Mixer for FL

This tutorial covers MLP-Mixer as an alternative to CNNs/Transformers.

## Architecture

- Patch-based MLP mixing.
- No attention mechanism.

## Benefits for FL

- Simpler, fewer parameters.
- Potentially better generalization.

## Configuration

```yaml
model:
  architecture: "mlp_mixer"
  patch_size: 16
  hidden_dim: 256
```

## Exercises

1. Comparison with ResNet in FL.
2. Token vs channel mixing.
