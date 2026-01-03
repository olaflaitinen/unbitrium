# Tutorial 107: Federated Time Series Forecasting

This tutorial covers FL for time series prediction.

## Setting

- Clients hold temporal data (IoT sensors, energy meters).
- Non-IID: Different seasonality patterns per client.

## Configuration

```yaml
model:
  type: "transformer"
  d_model: 64
  nhead: 4

data:
  lookback: 24
  horizon: 6
```

## Exercises

1. Handling missing timestamps across clients.
2. Impact of temporal misalignment on global model.
