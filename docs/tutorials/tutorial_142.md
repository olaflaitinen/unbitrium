# Tutorial 142: Scaling FL Simulations

This tutorial covers scaling to large client populations.

## Strategies

- **Parallelization**: Multi-process client simulation.
- **Sampling**: Simulate only active clients.
- **Caching**: Reuse partitioned data.

## Configuration

```yaml
simulation:
  parallel_workers: 8
  client_caching: true
```

## Exercises

1. Memory management for large simulations.
2. Trade-offs in sampling strategies.
