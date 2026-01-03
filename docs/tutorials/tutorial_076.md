# Tutorial 076: Gradient Compression (Quantization)

## Overview
Reducing bandwidth by quantization.

## Method
**Stochastic Quantization**: $Q(x) = \|x\| \text{sign}(x) \xi(x, s)$.

## Code
```python
compressor = ub.compression.StochasticQuantization(num_levels=256) # 8-bit
```

## Result
Model size reduced $4\times$ (32-bit $\to$ 8-bit) with negligible accuracy loss.
