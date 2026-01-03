# Tutorial 191: Transformer Architectures for FL

This tutorial covers Transformer models in FL.

## Considerations

- Very large models (BERT, GPT).
- High communication cost.
- Memory requirements.

## Strategies

- LoRA and adapter layers.
- Federated distillation.
- Partial parameter sharing.

## Configuration

```yaml
model:
  architecture: "bert-base"
  adaptation: "lora"
  lora_rank: 8
```

## Exercises

1. Fine-tuning vs full training.
2. Parameter-efficient FL for LLMs.
