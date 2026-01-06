"""Network simulation for federated learning.

Provides network topology and latency modeling.

Author: Olaf Yunus Laitinen Imanov <oyli@dtu.dk>
License: EUPL-1.2
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class NetworkConfig:
    """Network configuration for simulation.

    Args:
        latency_mean: Mean latency in milliseconds.
        latency_std: Latency standard deviation.
        packet_loss_rate: Probability of packet loss [0, 1].
        bandwidth_mbps: Bandwidth in Mbps.
    """
    latency_mean: float = 100.0
    latency_std: float = 20.0
    packet_loss_rate: float = 0.0
    bandwidth_mbps: float = 10.0

    def simulate_latency(self) -> float:
        """Simulate latency based on config."""
        return max(0, np.random.normal(self.latency_mean, self.latency_std))

    def simulate_packet_loss(self) -> bool:
        """Simulate packet loss."""
        return np.random.random() < self.packet_loss_rate


class Network:
    """Simulated network for federated learning.

    Args:
        num_clients: Number of clients.
        bandwidth_range: (min, max) bandwidth in Mbps.
        latency_range: (min, max) latency in ms.
        drop_rate: Probability of message drop.
        seed: Random seed.

    Example:
        >>> network = Network(num_clients=10)
        >>> delay = network.get_transmission_time(client_id=0, size_bytes=1024)
    """

    def __init__(
        self,
        num_clients: int,
        bandwidth_range: tuple[float, float] = (1.0, 100.0),
        latency_range: tuple[float, float] = (10.0, 200.0),
        drop_rate: float = 0.0,
        seed: int = 42,
    ) -> None:
        """Initialize network simulation.

        Args:
            num_clients: Number of clients.
            bandwidth_range: Bandwidth range in Mbps.
            latency_range: Latency range in ms.
            drop_rate: Message drop probability.
            seed: Random seed.
        """
        self.num_clients = num_clients
        self.drop_rate = drop_rate
        self.rng = np.random.default_rng(seed)

        # Assign random network properties per client
        self.bandwidths = self.rng.uniform(
            bandwidth_range[0], bandwidth_range[1], size=num_clients
        )
        self.latencies = self.rng.uniform(
            latency_range[0], latency_range[1], size=num_clients
        )

    def get_transmission_time(
        self,
        client_id: int,
        size_bytes: int,
    ) -> float:
        """Compute transmission time for a client.

        Args:
            client_id: Client identifier.
            size_bytes: Size of data in bytes.

        Returns:
            Transmission time in seconds.
        """
        bandwidth_bps = self.bandwidths[client_id] * 1e6 / 8  # Convert Mbps to bytes/s
        latency_s = self.latencies[client_id] / 1000  # Convert ms to seconds

        return latency_s + size_bytes / bandwidth_bps

    def is_message_dropped(self, client_id: int) -> bool:
        """Check if a message should be dropped.

        Args:
            client_id: Client identifier.

        Returns:
            True if message should be dropped.
        """
        return self.rng.random() < self.drop_rate

    def simulate_round_delays(
        self,
        model_size_bytes: int,
    ) -> dict[int, float]:
        """Simulate transmission delays for all clients.

        Args:
            model_size_bytes: Size of model in bytes.

        Returns:
            Dictionary mapping client ID to delay in seconds.
        """
        delays = {}
        for client_id in range(self.num_clients):
            if not self.is_message_dropped(client_id):
                delays[client_id] = self.get_transmission_time(
                    client_id, model_size_bytes
                )
        return delays
