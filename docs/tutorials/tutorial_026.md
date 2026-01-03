# Tutorial 026: FedCM (Client Momentum)

## Overview
**FedCM** uses momentum on the client side to smooth out local updates, reducing the variance of the gradients sent to the server.

## Experiment
Compare FedAvg vs FedCM on a highly noisy dataset (e.g., small batch size).

## Code

```python
agg = ub.aggregators.FedCM(beta=0.9)
```

## Observation
FedCM helps when local batch sizes are small (high stochastic noise).
