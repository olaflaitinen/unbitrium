# Tutorial 159: Gossip-based FL

This tutorial covers decentralized FL using gossip protocols.

## Algorithm

Clients exchange models with neighbors, no central server.

## Configuration

```yaml
topology:
  type: "gossip"
  mixing_matrix: "metropolis"
  neighbors: 3
```

## Exercises

1. Graph topology impact on convergence.
2. Bandwidth savings vs star topology.
