# Tutorial 055: Model Pruning Setup

## Overview
Simulating the effect of pruning (sparsification) on communication cost.

## Masking
We generate a binary mask $M \in \{0, 1\}$ for weights. Only send non-zero weights.

## Experiment
Vary sparsity from 0% to 99%.

## Impact
- **Comm Cost**: Linear reduction.
- **Accuracy**: Non-linear drop (Lottery Ticket Hypothesis).

## Code
```python
compressor = ub.compression.Pruning(sparsity=0.9)
compressed_update = compressor.compress(weights)
```
