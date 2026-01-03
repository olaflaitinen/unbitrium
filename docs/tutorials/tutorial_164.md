# Tutorial 164: Mixup and Data Augmentation in FL

This tutorial covers data augmentation strategies for FL.

## Techniques

- **Mixup**: Convex combinations of samples.
- **Cutout**: Random masking.
- **AutoAugment**: Learned policies.

## Configuration

```yaml
augmentation:
  - type: "mixup"
    alpha: 0.4
  - type: "random_crop"
    padding: 4
```

## Exercises

1. Federated learning of augmentation policies.
2. Impact on heterogeneity mitigation.
