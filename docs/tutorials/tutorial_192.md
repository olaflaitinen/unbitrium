# Tutorial 192: GAN Architectures for FL

This tutorial covers GAN training in FL.

## Challenges

- Mode collapse across clients.
- Non-convergent dynamics.

## Strategies

- MD-GAN: Multiple discriminators.
- FEGAN: Feature-aligned aggregation.

## Configuration

```yaml
model:
  type: "dcgan"
  latent_dim: 100
```

## Exercises

1. Evaluating federated GANs (FID).
2. Privacy of generated samples.
