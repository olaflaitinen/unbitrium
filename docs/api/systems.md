# Systems API Reference

This document provides the API reference for the `unbitrium.systems` module.

---

## Table of Contents

1. [Device](#device)
2. [EnergyModel](#energymodel)
3. [Network](#network)

---

## Device

```python
from unbitrium.systems import Device

@dataclass
class Device:
    """Simulated edge device.

    Args:
        device_id: Unique identifier.
        compute_power: FLOPS capacity.
        memory: RAM in bytes.
        battery: Battery capacity mAh.

    Example:
        >>> device = Device(
        ...     device_id=0,
        ...     compute_power=1e9,
        ...     memory=4e9,
        ...     battery=3000,
        ... )
        >>> train_time = device.estimate_training_time(model, num_samples)
    """

    device_id: int
    compute_power: float
    memory: float
    battery: float
```

---

## EnergyModel

```python
from unbitrium.systems import EnergyModel

@dataclass
class EnergyModel:
    """Energy consumption model.

    Args:
        compute_power_watts: Power during computation.
        idle_power_watts: Idle power consumption.
        comm_power_watts: Communication power.
    """

    compute_power_watts: float = 5.0
    idle_power_watts: float = 0.5
    comm_power_watts: float = 2.0
```

---

## Network

```python
from unbitrium.simulation import Network

class Network:
    """Network simulation with latency and loss.

    Args:
        bandwidth: Bandwidth in bytes/second.
        latency_mean: Mean latency in seconds.
        latency_std: Latency standard deviation.
        drop_rate: Packet drop probability.
    """
```

---

*Last updated: January 2026*
