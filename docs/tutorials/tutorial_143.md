# Tutorial 143: Reproducibility Best Practices

This tutorial covers ensuring FL experiment reproducibility.

## Checklist

1. Set all random seeds.
2. Use deterministic algorithms.
3. Version lock dependencies.
4. Capture provenance manifests.

## Configuration

```yaml
reproducibility:
  seed: 42
  deterministic: true
  save_manifest: true
```

## Exercises

1. Debugging non-determinism.
2. Reproducing across hardware.
