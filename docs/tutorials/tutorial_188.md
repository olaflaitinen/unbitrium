# Tutorial 188: Medical Imaging Datasets

This tutorial covers medical imaging for FL.

## Datasets

- ChestX-ray14.
- CheXpert.
- ISIC skin lesion.

## Configuration

```yaml
dataset:
  name: "chexpert"
  split_by: "hospital"
```

## Challenges

- Label noise.
- Domain shift across institutions.
- Regulatory compliance.

## Exercises

1. Federated pretrain, local finetune.
2. Multi-task learning across conditions.
