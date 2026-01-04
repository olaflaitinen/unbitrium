"""Device and energy models for Unbitrium.

Provides simulation primitives for heterogeneous device capabilities.

Author: Olaf Yunus Laitinen Imanov <oyli@dtu.dk>
License: EUPL-1.2
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class Device:
    """Simulated edge device for federated learning.

    Attributes:
        device_id: Unique device identifier.
        compute_capacity: Relative compute speed (1.0 = baseline).
        memory_mb: Available memory in megabytes.
        bandwidth_mbps: Network bandwidth in Mbps.
        battery_mah: Battery capacity in mAh.
        is_available: Whether device is currently available.
    """

    device_id: int
    compute_capacity: float = 1.0
    memory_mb: int = 4096
    bandwidth_mbps: float = 10.0
    battery_mah: int = 3000
    is_available: bool = True

    def compute_training_time(
        self,
        model_flops: int,
        num_samples: int,
        batch_size: int,
    ) -> float:
        """Estimate training time for given workload.

        Args:
            model_flops: FLOPs per forward/backward pass.
            num_samples: Number of training samples.
            batch_size: Training batch size.

        Returns:
            Estimated training time in seconds.
        """
        num_batches = (num_samples + batch_size - 1) // batch_size
        total_flops = model_flops * num_batches * batch_size
        # Assume baseline device does 1e9 FLOPS
        baseline_flops_per_second = 1e9
        effective_flops = baseline_flops_per_second * self.compute_capacity
        return total_flops / effective_flops

    def compute_communication_time(self, model_size_bytes: int) -> float:
        """Estimate communication time for model upload/download.

        Args:
            model_size_bytes: Model size in bytes.

        Returns:
            Estimated communication time in seconds.
        """
        bytes_per_second = self.bandwidth_mbps * 1e6 / 8
        return model_size_bytes / bytes_per_second


@dataclass
class EnergyModel:
    """Energy consumption model for federated learning.

    Attributes:
        compute_power_w: Power consumption during computation (Watts).
        idle_power_w: Power consumption when idle (Watts).
        transmit_power_w: Power consumption during transmission (Watts).
    """

    compute_power_w: float = 2.5
    idle_power_w: float = 0.5
    transmit_power_w: float = 1.5

    def compute_energy(
        self,
        compute_time: float,
        communication_time: float,
        idle_time: float = 0.0,
    ) -> float:
        """Compute total energy consumption.

        Args:
            compute_time: Time spent computing (seconds).
            communication_time: Time spent communicating (seconds).
            idle_time: Time spent idle (seconds).

        Returns:
            Total energy consumption in Joules.
        """
        return (
            self.compute_power_w * compute_time
            + self.transmit_power_w * communication_time
            + self.idle_power_w * idle_time
        )

    def estimate_battery_drain(
        self,
        energy_j: float,
        battery_mah: int,
        voltage: float = 3.7,
    ) -> float:
        """Estimate battery drain percentage.

        Args:
            energy_j: Energy consumed in Joules.
            battery_mah: Battery capacity in mAh.
            voltage: Battery voltage (default: 3.7V for Li-ion).

        Returns:
            Battery drain as percentage (0-100).
        """
        battery_wh = battery_mah * voltage / 1000
        battery_j = battery_wh * 3600
        return (energy_j / battery_j) * 100
