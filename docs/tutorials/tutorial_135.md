# Tutorial 135: Federated Keyboard Prediction

This tutorial covers FL for on-device text prediction (Gboard).

## Setting

- Mobile devices as clients.
- Next-word prediction or suggestion.
- Highly privacy-sensitive (keystrokes).

## Architecture

- LSTM or Transformer LM.
- On-device training with differential privacy.

## Configuration

```yaml
domain: "keyboard"
model:
  type: "lstm"
  vocab_size: 10000

privacy:
  clip_norm: 1.0
  noise_multiplier: 0.4
```

## Exercises

1. Handling multilingual users.
2. Personalization vs global model balance.
