# Tutorial 117: Federated Generative Models

This tutorial covers FL for GANs and VAEs.

## Approach

- Federated GAN: Clients train discriminators locally.
- Federated VAE: Aggregate encoder/decoder separately.

## Configuration

```yaml
model:
  type: "vae"
  latent_dim: 32

training:
  kl_weight: 0.01
```

## Exercises

1. Mode collapse in federated GANs.
2. Privacy of generated samples.
