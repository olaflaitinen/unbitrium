# Tutorial 097: Carbon Footprint Estimation

This tutorial adds carbon tracking to the simulator.

## Methodology

$$
CO_2 = (\text{Energy}_{train} + \text{Energy}_{comm}) \times \text{CarbonIntensity}
$$

## Implementation

Extend `EnergyModel` in `unbitrium.systems` to multiply by intensity factors (gCO2/kWh) per region.

## Exercises

1. Compare Carbon footprint of FL vs Centralized training.
2. Schedule clients based on green energy availability.
