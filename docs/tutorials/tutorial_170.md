# Tutorial 170: Model Checkpointing in FL

This tutorial covers saving and loading federated checkpoints.

## Best Practices

- Save global model at intervals.
- Store round number and metadata.
- Optionally save client states.

## Code

```python
checkpoint = {
    'round': current_round,
    'model_state': global_model.state_dict(),
    'config': config,
    'metrics': history
}
torch.save(checkpoint, f'checkpoint_r{current_round}.pt')
```

## Exercises

1. Resume training from checkpoint.
2. Checkpoint formats for interoperability.
