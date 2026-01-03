# Tutorial 146: Statistical Testing for FL

This tutorial covers statistical significance testing for FL experiments.

## Tests

- Paired t-test for comparing aggregators.
- Bootstrap confidence intervals.
- Effect size (Cohen's d).

## Code

```python
from scipy import stats

# Compare FedAvg vs FedProx accuracies
t_stat, p_value = stats.ttest_rel(fedavg_accs, fedprox_accs)
print(f"p-value: {p_value:.4f}")
```

## Exercises

1. Multiple comparison correction (Bonferroni).
2. Non-parametric alternatives.
