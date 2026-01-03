# Tutorial 068: Text Data (NLP)

## Overview
Federated Language Modeling (Next Word Prediction).

## Model
LSTM/Transformer.

## Dataset
Shakespeare (Character level) or StackOverflow (Word level).

## Batching
Text batches have variable sequence lengths. Need a custom `collate_fn` in the client standard loader.

## Code
```python
engine = ub.core.SimulationEngine(..., collate_fn=pad_sequence)
```
