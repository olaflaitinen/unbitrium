# Tutorial 048: Hierarchical FL (Cross-Silo)

## Overview
Silos (Cities) $\rightarrow$ Server (Country).
Inside each Silo, there are Devices.

## Mockup
Run 2 instances of Unbitrium engine as "Silo Aggregators", then feed their outputs to a "Master Aggregator".

## Code
```python
# Conceptual
# Silo 1
w1 = engine1.run()
# Silo 2
w2 = engine2.run()
# Global
w_global = (w1 + w2) / 2
```
