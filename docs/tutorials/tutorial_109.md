# Tutorial 109: Federated Semantic Segmentation

This tutorial covers FL for pixel-wise classification.

## Setting

- Clients hold segmentation masks (autonomous driving, medical imaging).
- High communication cost due to large models.

## Configuration

```yaml
model:
  type: "unet"
  encoder: "resnet34"

training:
  gradient_accumulation: 4
```

## Exercises

1. Strategies to reduce communication for large models.
2. Handling label noise in segmentation masks.
