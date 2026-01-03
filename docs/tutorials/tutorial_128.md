# Tutorial 128: Model Heterogeneity in FL

This tutorial covers FL when clients have different model architectures.

## Approaches

- **Knowledge Distillation**: Exchange logits, not weights.
- **FedMD**: Match predictions on public dataset.
- **Submodel Training**: Smaller clients train subsets.

## Configuration

```yaml
model_heterogeneity:
  method: "fedmd"
  public_dataset_size: 5000
```

## Exercises

1. Privacy of shared predictions.
2. Handling capability gaps.
