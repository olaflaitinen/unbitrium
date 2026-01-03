# Tutorial 158: FedDC

This tutorial covers FedDC (Federated Daisy Chaining).

## Concept

Sequential model passing between clients for local refinement.

## Configuration

```yaml
aggregator:
  type: "feddc"
  chain_length: 5
```

## Exercises

1. Privacy implications of sequential passing.
2. Convergence analysis.
