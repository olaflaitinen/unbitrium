# Tutorial 144: Configuration Management

This tutorial covers managing FL experiment configurations.

## Best Practices

- Use YAML for human readability.
- Schema validation.
- Hierarchical defaults with overrides.
- Version control configurations.

## Example

```yaml
defaults:
  - dataset: cifar10
  - aggregator: fedavg

experiment:
  seed: ${seed:42}
  rounds: 100
```

## Exercises

1. Hydra-style configuration composition.
2. Environment-specific overrides.
