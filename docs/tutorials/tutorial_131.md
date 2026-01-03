# Tutorial 131: Federated Healthcare Applications

This tutorial covers FL applications in healthcare.

## Use Cases

- Collaborative model training across hospitals.
- Rare disease detection with limited samples per site.
- Multi-institutional clinical trials.

## Privacy Requirements

- HIPAA compliance.
- Differential privacy often mandatory.

## Configuration

```yaml
domain: "healthcare"
privacy:
  mechanism: "gaussian"
  epsilon: 8.0
  delta: 1e-5
```

## Exercises

1. Data harmonization across different EHR systems.
2. Regulatory considerations for cross-border FL.
