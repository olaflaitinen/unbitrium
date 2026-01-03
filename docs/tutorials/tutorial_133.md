# Tutorial 133: Federated IoT and Edge Computing

This tutorial covers FL for IoT deployments.

## Setting

- Sensors, gateways, edge devices.
- Extreme resource constraints.
- Intermittent connectivity.

## Configuration

```yaml
domain: "iot"
device:
  memory_mb: 64
  compute_mflops: 100

network:
  bandwidth_kbps: 128
  availability: 0.7
```

## Challenges

- Model quantization for edge deployment.
- Handling device churn.

## Exercises

1. Trade-offs in model compression for FL.
2. Offline-first training strategies.
