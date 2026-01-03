# Tutorial 127: Layer-wise Aggregation Strategies

This tutorial covers selective layer aggregation.

## Approaches

- Aggregate only classifier head.
- Aggregate base, personalize head.
- Freeze embeddings, aggregate rest.

## Configuration

```yaml
aggregation:
  layers:
    - name: "classifier"
      aggregate: true
    - name: "backbone.*"
      aggregate: false
```

## Exercises

1. Which layers benefit most from aggregation?
2. Communication savings from partial aggregation.
