"""Simulation package for Unbitrium.

Provides federated learning simulation infrastructure including
client, server, and network modeling.

Author: Olaf Yunus Laitinen Imanov <oyli@dtu.dk>
License: EUPL-1.2
"""

from __future__ import annotations

from unbitrium.simulation.client import Client
from unbitrium.simulation.server import Server
from unbitrium.simulation.network import Network
from unbitrium.simulation.simulator import Simulator

__all__ = [
    "Client",
    "Server",
    "Network",
    "Simulator",
]
