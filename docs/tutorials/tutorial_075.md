# Tutorial 075: Multi-Node Simulation (Ray)

## Overview
Scaling to clusters for massive simulations (1000+ clients). Unbitrium supports Ray as a backend.

## Prerequisite
`pip install ray`

## Code
```python
import ray
ray.init(address="auto")

config = ub.core.SimulationConfig(backend="ray")
engine = ub.core.SimulationEngine(config, ...)
```

## Architecture
- **Driver**: Server logic.
- **Workers**: Ray Actors (Remote Clients).
