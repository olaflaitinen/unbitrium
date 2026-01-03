# Tutorial 032: Packet Loss and Reselection

## Overview
Simulating packet loss (drop out). Use `SimulationEngine` to simulate client failure probabilities.

## Strategy
If a client drops out, does the server wait (timeout) or proceed?

## Experiment
Set `packet_loss_rate=0.1` (10%).

## Code
```python
# Simulating dropout in the engine loop
passed = np.random.rand() > packet_loss_rate
if not passed:
    # Client fails to return update
    pass
```

## Impact
Effective batch size decreases. If dropouts are correlated with data properties (e.g., clients with "Hard" images take longer -> timeout), we get bias.
