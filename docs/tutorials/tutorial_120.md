# Tutorial 120: Federated Bayesian Learning

This tutorial covers Bayesian approaches to FL.

## Approach

- Posterior aggregation across clients.
- Uncertainty quantification via ensembles or variational inference.

## Configuration

```yaml
bayesian:
  method: "variational"
  prior: "gaussian"
  kl_weight: 1e-4
```

## Exercises

1. Aggregating posterior distributions.
2. Federated Gaussian Processes.
