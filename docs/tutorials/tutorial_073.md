# Tutorial 073: Using PyTorch Profiler

## Overview
Identifying GPU bottlenecks or data loading lag.

## Code
```python
with torch.profiler.profile(...) as p:
    engine.run()

print(p.key_averages().table(sort_by="cuda_time_total"))
```

## Common Bottleneck
In FL simulation, **Model Loading** (moving weights CPU $\to$ GPU) is often the bottleneck, not the Forward/Backward pass, because we swap context every few seconds.
