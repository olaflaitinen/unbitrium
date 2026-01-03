# Tutorial 106: Federated Natural Language Processing

This tutorial covers FL for NLP tasks (next word prediction, sentiment).

## Setting

- Clients hold text corpora (emails, messages).
- Privacy-sensitive content must not leave device.

## Configuration

```yaml
model:
  type: "lstm"
  embedding_dim: 200
  hidden_dim: 256

tokenizer:
  vocab_size: 10000
```

## Exercises

1. Handling OOV tokens across clients.
2. Federated fine-tuning of pretrained language models.
