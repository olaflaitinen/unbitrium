# Tutorial 123: Straggler Mitigation Techniques

This tutorial covers handling slow clients.

## Techniques

- **Timeout**: Drop clients exceeding deadline.
- **Oort**: Predict completion time, select accordingly.
- **Partial Updates**: Accept incomplete training.

## Configuration

```yaml
stragglers:
  timeout_ms: 10000
  policy: "partial_update"
```

## Exercises

1. Trade-off between completeness and speed.
2. Bias from dropping slow clients.
