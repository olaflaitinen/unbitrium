# Tutorial 155: Bulyan Aggregation

This tutorial covers Bulyan for nested robust aggregation.

## Algorithm

1. Use Krum to select $n-2f$ candidates.
2. Apply trimmed mean on candidates.

## Configuration

```yaml
aggregator:
  type: "bulyan"
  num_byzantine: 2
```

## Exercises

1. When Bulyan outperforms Krum.
2. Computational complexity.
