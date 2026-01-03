# Tutorial 014: Label Entropy vs Accuracy

## Overview
Does the "purity" of a client's dataset (low entropy) make local training easier or harder?

## Hypothesis
- **Low Entropy (1 class)**: Local model achieves 100% accuracy quickly (overfitting to 1 class), but 10% on global test.
- **High Entropy (balanced)**: Local model learns general features, decent global accuracy.

## Experiment
Correlate `compute_label_entropy(client_data)` with the client's contribution to global accuracy gain.

## Code

```python
import unbitrium as ub
from scipy.stats import pearsonr

# ... Setup partition ...
entropies = []
accuracies = [] # collected from sim

# Analysis
corr, _ = pearsonr(entropies, accuracies)
print(f"Entropy-Accuracy Correlation: {corr}")
```

## Result
Usually a positive correlation for FedAvg: clients with higher entropy (more diverse data) produce updates that generalize better.
