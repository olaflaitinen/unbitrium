# Tutorial 031: Network Latency Impact

## Overview
How does network latency (RTT) affect the total training time? In synchronous FL, the round time is determined by the slowest client.

## Experiment
Vary average RTT from 50ms (typical) to 2000ms (satellite).

## Code

```python
import unbitrium as ub

network = ub.systems.NetworkModel()
stats = network.sample_stats() # Returns object with latency_ms

# Use stats in cost model
# ...
```

## Result
Total Time = Rounds $\times$ (Compute + Comm).
If Comm dominates, investigating compression (Quantization) becomes priority.
