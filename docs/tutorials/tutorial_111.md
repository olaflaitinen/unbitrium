# Tutorial 111: Semi-Supervised Federated Learning

This tutorial covers FL with limited labeled data.

## Approach

- Use pseudo-labeling or contrastive learning on unlabeled data.
- Mix labeled and unlabeled losses.

## Configuration

```yaml
semi_supervised:
  method: "pseudo_label"
  confidence_threshold: 0.95
  unlabeled_weight: 0.5
```

## Exercises

1. Quality of pseudo-labels under non-IID.
2. Federated contrastive learning approaches.
