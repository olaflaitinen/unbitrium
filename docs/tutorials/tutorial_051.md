# Tutorial 051: Introduction to Personalized FL (FedPer)

## Overview
FedPer (Arivazhagan et al., 2019) splits the model into Base (Global) and Head (Personal/Local).

## Concept
- **Base Layers**: Aggregated using FedAvg. Learn universal features.
- **Head Layers**: Kept local, never sent to server. Adapt to local labels.

## Implementation Details
We define the cut point in the model architecture.

```python
# Pseudo-code config
config = ub.core.SimulationConfig(
    personalization_strategy="FedPer",
    split_layer="fc1" # last fully connected layer starts head
)
```

## Benefits
Accurate on local data (personalization), robust to label skew (heads adapt to missing classes).
