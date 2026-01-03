# Tutorial 035: Energy Constraint Simulation

## Overview
Battery-powered clients stop participating when battery < 20%.

## Simulation
Track `battery_level` per client. Decrement after each round based on `estimate_energy`.

## Code

```python
joules_per_round = ub.metrics.estimate_energy(flops=1e9)
client_battery -= joules_per_round

if client_battery < 20.0:
    is_available = False
```

## Long-term dynamics
Eventually, only plugged-in devices (bias?) remain in the cohort.
