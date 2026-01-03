# Tutorial 140: Debugging FL Experiments

This tutorial covers debugging techniques for FL.

## Common Issues

1. **Divergence**: Check for NaN gradients, LR too high.
2. **No Convergence**: Verify data loading, aggregation.
3. **Poor Generalization**: Check non-IID severity.

## Tools

- TensorBoard for loss curves.
- Gradient histograms.
- Weight drift tracking.

## Configuration

```yaml
debug:
  log_level: "DEBUG"
  save_client_updates: true
  gradient_histograms: true
```

## Exercises

1. Diagnosing client-level issues.
2. Using provenance manifests for reproducibility.
