# Tutorial 195: Efficient Networks for Cross-Device FL

This tutorial covers efficient architectures for mobile FL.

## Architectures

- MobileNet-V2/V3.
- EfficientNet.
- ShuffleNet.

## Configuration

```yaml
model:
  architecture: "mobilenetv3_small"
  width_mult: 0.5
```

## Exercises

1. MACs vs accuracy trade-off.
2. Quantized training on device.
