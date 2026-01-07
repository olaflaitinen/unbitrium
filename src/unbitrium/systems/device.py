"""Device and energy models for Unbitrium.

Provides simulation primitives for heterogeneous device capabilities.

Author: Olaf Yunus Laitinen Imanov <oyli@dtu.dk>
License: EUPL-1.2
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class EnergyModel:
    """Energy consumption model for federated learning.

    Attributes:
        joules_per_gflop: Energy per gigaflop of computation.
        joules_per_mb: Energy per megabyte of communication.
        compute_power_w: Power consumption during computation (Watts).
        idle_power_w: Power consumption when idle (Watts).
        transmit_power_w: Power consumption during transmission (Watts).
    """

    joules_per_gflop: float = 0.01
    joules_per_mb: float = 0.5
    compute_power_w: float = 2.5
    idle_power_w: float = 0.5
    transmit_power_w: float = 1.5

    def compute_energy_joules(self, gflops: float) -> float:
        """Compute energy from GFLOPs.

        Args:
            gflops: Number of gigaflops.

        Returns:
            Energy in joules.
        """
        return gflops * self.joules_per_gflop

    def estimate_training_energy(
        self,
        num_samples: int,
        batch_size: int = 32,
        model_flops_per_sample: float = 1e6,
    ) -> float:
        """Estimate energy for training.

        Args:
            num_samples: Number of training samples.
            batch_size: Training batch size.
            model_flops_per_sample: FLOPs per sample.

        Returns:
            Estimated energy in joules.
        """
        total_flops = num_samples * model_flops_per_sample
        gflops = total_flops / 1e9
        return self.compute_energy_joules(gflops)

    def estimate_communication_energy(self, size_mb: float) -> float:
        """Estimate energy for communication.

        Args:
            size_mb: Data size in megabytes.

        Returns:
            Estimated energy in joules.
        """
        return size_mb * self.joules_per_mb

    def estimate_round_energy(
        self,
        training_samples: int,
        model_size_mb: float,
        batch_size: int = 32,
        model_flops_per_sample: float = 1e6,
    ) -> float:
        """Estimate total energy for one FL round.

        Args:
            training_samples: Number of training samples.
            model_size_mb: Model size in megabytes.
            batch_size: Training batch size.
            model_flops_per_sample: FLOPs per sample.

        Returns:
            Total estimated energy in joules.
        """
        training_energy = self.estimate_training_energy(
            training_samples, batch_size, model_flops_per_sample
        )
        comm_energy = self.estimate_communication_energy(model_size_mb * 2)  # up + down
        return training_energy + comm_energy

    def battery_drain_percentage(
        self,
        energy_joules: float,
        battery_capacity_wh: float,
    ) -> float:
        """Calculate battery drain percentage.

        Args:
            energy_joules: Energy consumed in joules.
            battery_capacity_wh: Battery capacity in watt-hours.

        Returns:
            Battery drain as percentage (0-100).
        """
        battery_joules = battery_capacity_wh * 3600
        return (energy_joules / battery_joules) * 100


@dataclass
class Device:
    """Simulated edge device for federated learning.

    Attributes:
        device_id: Unique device identifier.
        compute_capacity: Relative compute speed (1.0 = baseline).
        memory_mb: Available memory in megabytes.
        bandwidth_mbps: Network bandwidth in Mbps.
        battery_level: Battery level as fraction (0.0-1.0).
        energy_model: Optional energy model for the device.
    """

    device_id: int = 0
    compute_capacity: float = 1.0
    memory_mb: int = 4096
    bandwidth_mbps: float = 10.0
    battery_level: float = 1.0
    energy_model: Any = field(default=None)

    def estimate_training_time(
        self,
        num_samples: int,
        model_flops_per_sample: float = 1e6,
        batch_size: int = 32,
    ) -> float:
        """Estimate training time for given workload.

        Args:
            num_samples: Number of training samples.
            model_flops_per_sample: FLOPs per forward/backward pass per sample.
            batch_size: Training batch size.

        Returns:
            Estimated training time in seconds.
        """
        total_flops = num_samples * model_flops_per_sample
        # Assume baseline device does 1e9 FLOPS
        baseline_flops_per_second = 1e9
        effective_flops = baseline_flops_per_second * self.compute_capacity
        return total_flops / effective_flops

    def can_fit_model(self, model_size_mb: float) -> bool:
        """Check if model fits in device memory.

        Args:
            model_size_mb: Model size in megabytes.

        Returns:
            True if model fits, False otherwise.
        """
        return model_size_mb <= self.memory_mb

    def estimate_upload_time(self, size_mb: float) -> float:
        """Estimate upload time for given data size.

        Args:
            size_mb: Data size in megabytes.

        Returns:
            Estimated upload time in seconds.
        """
        # Convert Mbps to MBps (megabytes per second)
        bandwidth_mbps_to_bytes = self.bandwidth_mbps / 8
        return size_mb / bandwidth_mbps_to_bytes

    def simulate_training(self, num_epochs: int, samples_per_epoch: int = 1000) -> None:
        """Simulate training and update battery level.

        Args:
            num_epochs: Number of training epochs.
            samples_per_epoch: Samples per epoch.
        """
        if self.energy_model is None:
            # Simple battery drain simulation
            drain_per_epoch = 0.05
            self.battery_level = max(
                0.0, self.battery_level - drain_per_epoch * num_epochs
            )
        else:
            energy = self.energy_model.estimate_training_energy(
                num_samples=samples_per_epoch * num_epochs
            )
            # Assume 10 Wh battery capacity
            drain_pct = self.energy_model.battery_drain_percentage(energy, 10.0) / 100
            self.battery_level = max(0.0, self.battery_level - drain_pct)

    def is_available(self) -> bool:
        """Check if device is available for training.

        Returns:
            True if device has sufficient battery.
        """
        return self.battery_level > 0.0

    def compute_communication_time(self, model_size_bytes: int) -> float:
        """Estimate communication time for model upload/download.

        Args:
            model_size_bytes: Model size in bytes.

        Returns:
            Estimated communication time in seconds.
        """
        bytes_per_second = self.bandwidth_mbps * 1e6 / 8
        return model_size_bytes / bytes_per_second
