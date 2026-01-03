# Tutorial 074: Multi-GPU Simulation (Data Parallel)

## Overview
Speeding up simulation by distributing clients across GPUs.

## Strategy
- GPU 0: Clients 0-4
- GPU 1: Clients 5-9

## Code
```python
config = ub.core.SimulationConfig(
    accelerator="cuda",
    devices=[0, 1]
)
# Engine automatically schedules clients to available devices
```
