# Tutorial 094: Knowledge Distillation in FL

This tutorial demonstrates using Knowledge Distillation (FedDistill) to reduce communication.

## Approach

- **Teacher**: Global model (on server or public dataset).
- **Student**: Client models.
- **Transfer**: Exchanging logits on public data instead of weights.

## Code Snippet

```python
def distillation_loss(student_logits, teacher_logits, T=2.0):
    soft_targets = torch.nn.functional.softmax(teacher_logits / T, dim=1)
    return torch.nn.functional.cross_entropy(student_logits / T, soft_targets)
```

## Exercises

1. Pros and cons of Data-Free Knowledge Distillation.
2. Requirements for the public dataset.
