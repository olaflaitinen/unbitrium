"""
Unbitrium: Federated Learning Simulator and Benchmarking Platform.

Copyright 2026 Olaf Yunus Laitinen Imanov and Contributors.
Licensed under the EUPL-1.2.
"""

from unbitrium.core.engine import SimulationEngine, SimulationConfig
from unbitrium.core.events import EventSystem

__version__ = "0.1.0"
__author__ = "Olaf Yunus Laitinen Imanov"
__license__ = "EUPL-1.2"

__all__ = [
    "SimulationEngine",
    "SimulationConfig",
    "EventSystem",
]
