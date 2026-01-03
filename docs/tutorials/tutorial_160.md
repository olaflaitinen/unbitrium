# Tutorial 160: D-SGD (Decentralized SGD)

This tutorial covers decentralized SGD without central coordination.

## Algorithm

$$
w_i^{t+1} = \sum_j W_{ij} (w_j^t - \eta \nabla F_j(w_j^t))
$$

## Configuration

```yaml
topology:
  type: "ring"

aggregator:
  type: "dsgd"
```

## Exercises

1. Spectral gap and convergence rate.
2. Handling dynamic topologies.
