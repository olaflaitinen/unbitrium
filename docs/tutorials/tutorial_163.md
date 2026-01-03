# Tutorial 163: SAM (Sharpness-Aware Minimization) in FL

This tutorial covers applying SAM for better generalization.

## Algorithm

Seek flat minima by perturbing towards sharpness.

$$
\epsilon^* = \rho \frac{\nabla L(w)}{||\nabla L(w)||}
$$

## Configuration

```yaml
client:
  optimizer: "sam"
  rho: 0.05
```

## Exercises

1. Communication cost of SAM in FL.
2. Impact on generalization under non-IID.
