"""Systems package for Unbitrium.

Provides device and energy modeling for federated learning simulations.

Author: Olaf Yunus Laitinen Imanov <oyli@dtu.dk>
License: EUPL-1.2
"""

from __future__ import annotations

from unbitrium.systems.device import Device, EnergyModel

__all__ = [
    "Device",
    "EnergyModel",
]
