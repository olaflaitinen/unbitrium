# Tutorial 115: Federated Neural Architecture Search

This tutorial covers NAS in FL settings.

## Approach

- Weight sharing across NAS supernet.
- Clients jointly search for optimal architecture.

## Configuration

```yaml
nas:
  search_space: "darts"
  warmup_epochs: 10
```

## Exercises

1. Architecture transfer across heterogeneous clients.
2. Communication cost of architecture parameters.
