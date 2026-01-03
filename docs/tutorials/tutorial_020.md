# Tutorial 020: Total Variation Distance (TVD)

## Overview
$TV(P, Q) = \frac{1}{2}\sum |p_i - q_i|$. A simple L1-based divergence metric.

## Comparison to EMD
- **TVD**: Ignores metric structure (Class 0 is as far from Class 1 as Class 9).
- **EMD**: Respects metric (if defined).

## Tutorial
Compute both TVD and EMD for a partition and discuss when to use which.
- Use **EMD** for ordinal data (Age, Severity).
- Use **TVD** for categorical data (Cats vs Dogs).

## Code
```python
val_tvd = 0.5 * np.sum(np.abs(p - q))
```
