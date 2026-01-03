# Tutorial 030: AFL-DCS (Asynchronous)

## Overview
**AFL-DCS** handles asynchronous updates. Clients push updates when ready; server aggregates immediately or buffers.

## Setup
Simulate clients with variable speed ($1\times, 0.5\times, 0.1\times$).

## Code

```python
config = ub.core.SimulationConfig(
    mode=ub.core.engine.SimulationMode.ASYNCHRONOUS,
    staleness_bound=20
)
agg = ub.aggregators.AFL_DCS()

engine = ub.core.SimulationEngine(config, agg)
engine.run()
```

## Outcome
Training finishes faster (wall-clock time) because we don't wait for stragglers. Accuracy might be slightly lower due to stale gradients.
