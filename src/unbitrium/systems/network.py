"""
Network simulation models.
"""

import numpy as np
from pydantic import BaseModel

class NetworkStats(BaseModel):
    bandwidth_mbps: float
    latency_ms: float
    packet_loss: float

class NetworkModel:
    """
    Simulates network conditions for clients.
    """

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)

    def sample_stats(self, region: str = "global") -> NetworkStats:
        """Samples network stats for a client."""
        # Simple Gaussian model for now
        bw = max(1.0, self.rng.normal(50, 20)) # Mbps
        lat = max(1.0, self.rng.normal(100, 30)) # ms
        loss = max(0.0, min(1.0, self.rng.exponential(0.01))) # rate

        return NetworkStats(
            bandwidth_mbps=bw,
            latency_ms=lat,
            packet_loss=loss
        )
