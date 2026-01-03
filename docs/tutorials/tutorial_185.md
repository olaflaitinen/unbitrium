# Tutorial 185: ImageNet Partitioning

This tutorial covers large-scale ImageNet for FL.

## Challenges

- Large dataset size.
- 1000-class imbalance.
- Compute requirements.

## Configuration

```yaml
dataset:
  name: "imagenet"
  subset: "imagenet100"  # Use 100-class subset

partitioning:
  strategy: "dirichlet"
  alpha: 0.3
```

## Exercises

1. Hierarchical label grouping.
2. Communication strategies for large models.
