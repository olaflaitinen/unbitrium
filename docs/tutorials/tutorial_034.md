# Tutorial 034: Device Compute Stragglers

## Overview
Clients have different compute capabilities (FLOPS). A high-end phone processes 1 batch in 10ms; a low-end IoT device takes 500ms.

## Code

```python
device_model = ub.systems.DeviceModel()
cap_high = device_model.sample_device("high")
cap_low = device_model.sample_device("low")

# Time = OPS / FLOPS
```

## Strategy
**FedProx** handles this by allowing partial work (variable epochs). The weak client does 1 epoch, strong does 10.
