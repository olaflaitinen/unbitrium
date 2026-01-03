# Tutorial 121: Cross-Device vs Cross-Silo FL

This tutorial compares the two main FL settings.

## Cross-Device

- Millions of clients (mobile phones).
- Small local datasets.
- High dropout rates.

## Cross-Silo

- Few clients (hospitals, banks).
- Large local datasets.
- Reliable participation.

## Configuration Differences

```yaml
# Cross-Device
clients: 10000
clients_per_round: 100
dropout_rate: 0.5

# Cross-Silo
clients: 5
clients_per_round: 5
dropout_rate: 0.0
```

## Exercises

1. Which aggregators suit each setting?
2. Privacy requirements comparison.
