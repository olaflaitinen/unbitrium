# Tutorial 194: Vision Transformers for FL

This tutorial covers ViT in federated settings.

## Architecture

- Patch embedding + Transformer.
- Self-attention over patches.

## Challenges

- Large model size.
- Data-hungry pretraining.

## Configuration

```yaml
model:
  architecture: "vit_small"
  patch_size: 16
  pretrained: true
```

## Exercises

1. Transfer learning with ViT in FL.
2. DeiT for data-efficient training.
