# Tutorial 154: Multi-Krum Aggregation

This tutorial covers Multi-Krum for improved robustness.

## Algorithm

Select $m$ best updates by Krum score, average them.

## Configuration

```yaml
aggregator:
  type: "multi_krum"
  num_select: 5
  num_byzantine: 2
```

## Exercises

1. Choosing $m$ given expected Byzantine fraction.
2. Trade-off between robustness and convergence.
