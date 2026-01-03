# Tutorial 102: Federated Transfer Learning

This tutorial covers using pretrained models in federated settings.

## Approach

- Freeze base layers, train only head.
- Reduces local computation and improves convergence when data is limited.

## Configuration

```yaml
model:
  backbone: "resnet18"
  pretrained: true
  freeze_until: "layer3"

training:
  fine_tune_epochs: 3
```

## Exercises

1. Impact of backbone choice on communication cost.
2. When to unfreeze layers progressively.
