from dataclasses import dataclass
from typing import Optional

@dataclass
class DeviceProfile:
    """
    Models the hardware constraints of a client device.
    """
    compute_capacity_flops: float  # FLOPS available
    memory_capacity_bytes: int     # RAM limit
    battery_capacity_mah: float    # Battery capacity
    current_battery_level: float   # 0.0 to 1.0
    energy_params: Optional['EnergyModel'] = None

    def can_participate(self, required_memory: int, required_flops: float) -> bool:
        if self.memory_capacity_bytes < required_memory:
            return False
        # Simple battery check
        if self.current_battery_level < 0.2:
            return False
        return True

    def estimate_training_time(self, flops_needed: float) -> float:
        return flops_needed / self.compute_capacity_flops

@dataclass
class EnergyModel:
    """
    Models energy consumption.
    """
    cpu_power_watts: float
    network_send_joules_per_bit: float
    network_recv_joules_per_bit: float

    def estimate_energy(self, computation_time: float, upload_bytes: int, download_bytes: int) -> float:
        compute_energy = computation_time * self.cpu_power_watts
        comm_energy = (upload_bytes * self.network_send_joules_per_bit +
                       download_bytes * self.network_recv_joules_per_bit)
        return compute_energy + comm_energy
