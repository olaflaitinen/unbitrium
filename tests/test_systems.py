"""Unit tests for systems module.

Author: Olaf Yunus Laitinen Imanov <oyli@dtu.dk>
License: EUPL-1.2
"""

from __future__ import annotations

import pytest

from unbitrium.systems import Device, EnergyModel


class TestDevice:
    """Tests for Device simulation."""

    def test_init_default(self) -> None:
        """Test default device initialization."""
        device = Device()
        assert device is not None

    def test_init_with_params(self) -> None:
        """Test device with custom parameters."""
        device = Device(
            compute_capacity=2.0,
            memory_mb=1024,
            battery_level=0.8,
        )
        assert device.compute_capacity == 2.0
        assert device.memory_mb == 1024
        assert device.battery_level == 0.8

    def test_compute_capacity_affects_training_time(self) -> None:
        """Test compute capacity impacts training time."""
        device_fast = Device(compute_capacity=4.0)
        device_slow = Device(compute_capacity=1.0)

        time_fast = device_fast.estimate_training_time(num_samples=1000)
        time_slow = device_slow.estimate_training_time(num_samples=1000)

        assert time_fast < time_slow

    def test_memory_constraint_check(self) -> None:
        """Test memory constraint checking."""
        device = Device(memory_mb=512)

        # Small model should fit
        assert device.can_fit_model(model_size_mb=100)

        # Large model should not fit
        assert not device.can_fit_model(model_size_mb=1000)

    def test_battery_drain(self) -> None:
        """Test battery consumption during training."""
        device = Device(battery_level=1.0)
        initial_battery = device.battery_level

        device.simulate_training(num_epochs=5)

        assert device.battery_level < initial_battery

    def test_is_available(self) -> None:
        """Test device availability check."""
        device_charged = Device(battery_level=0.5)
        device_dead = Device(battery_level=0.0)

        assert device_charged.is_available()
        assert not device_dead.is_available()

    def test_network_bandwidth(self) -> None:
        """Test network bandwidth property."""
        device = Device(bandwidth_mbps=100.0)
        assert device.bandwidth_mbps == 100.0

    def test_estimate_upload_time(self) -> None:
        """Test upload time estimation."""
        device = Device(bandwidth_mbps=10.0)

        # 10 MB at 10 Mbps = 8 seconds
        time_s = device.estimate_upload_time(size_mb=10.0)
        assert time_s == pytest.approx(8.0, rel=0.1)


class TestEnergyModel:
    """Tests for energy consumption modeling."""

    def test_init(self) -> None:
        """Test energy model initialization."""
        model = EnergyModel()
        assert model is not None

    def test_compute_from_flops(self) -> None:
        """Test energy computation from FLOPS."""
        model = EnergyModel(joules_per_gflop=0.1)

        energy = model.compute_energy_joules(gflops=100)
        assert energy == pytest.approx(10.0)

    def test_compute_training_energy(self) -> None:
        """Test training energy estimation."""
        model = EnergyModel()

        energy = model.estimate_training_energy(
            num_samples=1000,
            batch_size=32,
            model_flops_per_sample=1e6,
        )

        assert energy > 0

    def test_compute_communication_energy(self) -> None:
        """Test communication energy estimation."""
        model = EnergyModel(joules_per_mb=0.5)

        energy = model.estimate_communication_energy(size_mb=10.0)
        assert energy == pytest.approx(5.0)

    def test_total_round_energy(self) -> None:
        """Test total energy for one FL round."""
        model = EnergyModel()

        energy = model.estimate_round_energy(
            training_samples=1000,
            model_size_mb=50,
        )

        assert energy > 0

    def test_battery_drain_percentage(self) -> None:
        """Test battery drain percentage calculation."""
        model = EnergyModel()

        drain = model.battery_drain_percentage(
            energy_joules=100,
            battery_capacity_wh=10.0,
        )

        # 100 J = 100/3600 Wh = 0.0278 Wh
        # 0.0278 / 10 = 0.278%
        assert drain > 0
        assert drain < 1.0


class TestSystemIntegration:
    """Integration tests for systems module."""

    def test_device_with_energy_model(self) -> None:
        """Test device using energy model."""
        energy_model = EnergyModel()
        device = Device(
            compute_capacity=2.0,
            battery_level=0.8,
            energy_model=energy_model,
        )

        initial_battery = device.battery_level
        device.simulate_training(num_epochs=3)

        assert device.battery_level < initial_battery

    def test_multiple_devices_different_capacities(self) -> None:
        """Test heterogeneous device fleet."""
        devices = [
            Device(compute_capacity=1.0, memory_mb=256),
            Device(compute_capacity=2.0, memory_mb=512),
            Device(compute_capacity=4.0, memory_mb=1024),
        ]

        times = [d.estimate_training_time(num_samples=1000) for d in devices]

        # Times should decrease with capacity
        assert times[0] > times[1] > times[2]


class TestModuleExports:
    """Test systems module exports."""

    def test_exports(self) -> None:
        """Test all expected exports exist."""
        from unbitrium import systems

        assert hasattr(systems, "Device")
        assert hasattr(systems, "EnergyModel")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
