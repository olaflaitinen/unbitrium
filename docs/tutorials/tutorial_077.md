# Tutorial 077: Top-K Sparsification

## Overview
Keep only the largest $K$ gradients. Accumulate the rest in a local residual buffer (Error Compensation).

## Memory
Clients must remember the residual.

## Code
```python
compressor = ub.compression.TopK(ratio=0.01) # Keep 1%
compressor.enable_error_compensation(True)
```
