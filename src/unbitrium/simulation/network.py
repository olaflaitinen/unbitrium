"""
Network Simulation.
"""

from dataclasses import dataclass
import random
from typing import Optional

@dataclass
class NetworkConfig:
    """
    Configuration for network simulation.
    """
    bandwidth_upload: float = 10.0 # Mbps
    bandwidth_download: float = 50.0 # Mbps
    latency_mean: float = 0.050 # Seconds (50ms)
    latency_std: float = 0.010 # Seconds (10ms)
    packet_loss_rate: float = 0.0 # 0% to 100%
    energy_cost_per_byte: float = 0.0 # Joules/Byte (for energy sim)

class NetworkSimulator:
    """
    Simulates network conditions for FL.
    """

    def __init__(self, config: NetworkConfig):
        self.config = config

    def simulate_transmission_time(self, size_bytes: int, upload: bool = True) -> Optional[float]:
        """
        Calculates time to transmit 'size_bytes'.
        Returns None if packet loss occurs.
        """
        # 1. Packet Loss
        if self.config.packet_loss_rate > 0:
            if random.random() < self.config.packet_loss_rate:
                return None # Dropped

        # 2. Bandwidth
        bandwidth = self.config.bandwidth_upload if upload else self.config.bandwidth_download
        bandwidth_bps = bandwidth * 1e6

        # Time = Size / Bandwidth
        transfer_time = (size_bytes * 8) / bandwidth_bps

        # 3. Latency
        # Sample log-normal or normal latency?
        # Using Normal clipped at 0 for simplicity, or just mean + noise
        latency = random.gauss(self.config.latency_mean, self.config.latency_std)
        latency = max(0.001, latency) # Minimum 1ms

        return transfer_time + latency

    def estimate_energy(self, size_bytes: int) -> float:
        return size_bytes * self.config.energy_cost_per_byte
