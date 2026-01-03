# Tutorial 043: Reproducibility on GPU

## Overview
GPU operations in PyTorch are non-deterministic by default (e.g., `atomicAdd`).

## Solution
Unbitrium's `RNGManager` sets:
```python
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

## Performance Cost
This slows down training by ~10-15% but guarantees bit-wise identical results.

## Tutorial
Run the same config twice on GPU and verify the final model hash matches.
