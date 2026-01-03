# Tutorial 069: Hugging Face Integration

## Overview
Using `transformers` models in Unbitrium.

## Code
```python
from transformers import AutoModelForSequenceClassification

def model_fn():
    return AutoModelForSequenceClassification.from_pretrained("bert-tiny")

engine = ub.core.SimulationEngine(..., model_fn=model_fn)
```

## Note
Weights are large (BERT-Large = hundreds of MB). Simulation might be slow on single GPU due to swapping.
