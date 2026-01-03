# Tutorial 033: Bandwidth Heterogeneity

## Overview
Some clients are on 5G (100 Mbps), others on 3G (2 Mbps).

## Straggler Effect
The "straggler" effect is massive here.

## Mitigation
- **pFed**: Personalization might allow small models for weak clients.
- **Quantization**: Send INT8 instead of FP32.

## Simulation
Use `NetworkModel` to assign tiers to clients.
