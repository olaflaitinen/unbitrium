# Tutorial 024: FedSim (Similarity Weighted Aggregation)

## Overview
Instead of weighting by sample size ($n_k/n$), **FedSim** weights by cosine similarity to the global model update.

## Hypothesis
Updates that align with the global direction are "good". Updates that are orthogonal or opposite (due to local skew) should be down-weighted.

## Code

```python
import unbitrium as ub

agg = ub.aggregators.FedSim() # Automatic similarity weighting
```

## Scenario
Use the "Pathological" partition (Tutorial 009). FedSim should outperform FedAvg by suppressing conflicting gradients.
