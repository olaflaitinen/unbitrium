# Tutorial 125: Local Epochs vs Local Batch Iterations

This tutorial clarifies the distinction and its impact.

## Definitions

- **Local Epochs**: Full passes over local data.
- **Local Iterations**: Fixed number of mini-batch steps.

## Trade-offs

| Setting | Pros | Cons |
|---------|------|------|
| Epochs | Adapts to data size | More drift for large datasets |
| Iterations | Uniform compute | Underfitting for small clients |

## Configuration

```yaml
training:
  mode: "epochs"  # or "iterations"
  value: 5
```

## Exercises

1. When to prefer iterations over epochs?
2. Interaction with FedProx mu parameter.
