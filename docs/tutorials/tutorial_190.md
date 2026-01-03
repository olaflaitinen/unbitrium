# Tutorial 190: CNN Architectures for FL

This tutorial covers CNN choices for FL.

## Considerations

- Model size vs communication.
- Computation cost per round.

## Common Choices

| Model | Parameters | Notes |
|-------|------------|-------|
| LeNet | 60K | Baseline |
| VGG-11 | 128M | Heavy |
| ResNet-18 | 11M | Standard |
| MobileNet-V2 | 3.4M | Efficient |

## Configuration

```yaml
model:
  architecture: "resnet18"
  pretrained: false
```

## Exercises

1. Model scaling laws in FL.
2. Efficient architectures for cross-device.
