# Tutorial 124: Adaptive Learning Rate in FL

This tutorial covers server-side and client-side learning rate adaptation.

## Approaches

- **Server LR Decay**: Reduce LR over rounds.
- **FedOpt**: Adam-style adaptation on server.
- **Client Warmup**: Gradual LR increase per round.

## Configuration

```yaml
optimizer:
  server:
    type: "adam"
    lr: 0.01
  client:
    lr_schedule: "cosine"
```

## Exercises

1. When to decay server vs client LR?
2. Impact of mismatch between client and server LR.
