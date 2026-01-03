# Tutorial 025: pFedSim for Personalization

## Overview
**pFedSim** decouples the model into shared body and personalized heads.

## Setup
- **Architecture**: CNN Body + Linear Head.
- **Aggregation**: Average bodies based on similarity; Keep heads local.

## Code

```python
# Assuming aggregators.pFedSim handles the layer splitting logic via config
agg = ub.aggregators.pFedSim()
```

## Metric
Measure **Personalized Accuracy** (accuracy of Client K's model on Client K's test set) rather than Global Accuracy.
