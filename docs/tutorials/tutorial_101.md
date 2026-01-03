# Tutorial 101: Hierarchical Federated Learning

This tutorial covers hierarchical FL (multi-tier aggregation).

## Background

- **Two-tier hierarchy**: Edge servers aggregate regionally, cloud aggregates globally.
- **Benefits**: Reduced communication latency and bandwidth to centralized server.

## Configuration

```yaml
hierarchy:
  tiers: 2
  regions: 5
  clients_per_region: 20
  local_rounds_per_global: 5
```

## Data Flow

```mermaid
graph TD
    C1[Client 1] --> E1[Edge 1]
    C2[Client 2] --> E1
    C3[Client 3] --> E2[Edge 2]
    E1 --> Cloud[Cloud Server]
    E2 --> Cloud
```

## Exercises

1. Communication cost comparison vs flat FL.
2. Convergence analysis under hierarchical aggregation.
