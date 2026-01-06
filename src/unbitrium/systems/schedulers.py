"""
Client schedulers.
"""

from typing import List

import numpy as np


class ClientScheduler:
    """
    Policies for selecting clients (Random, Straggler-aware, etc.).
    """

    def __init__(self, num_clients: int, seed: int = 42):
        self.num_clients = num_clients
        self.rng = np.random.default_rng(seed)

    def select(self, num_to_select: int, round_num: int) -> List[int]:
        """Random selection."""
        return list(
            self.rng.choice(self.num_clients, size=num_to_select, replace=False)
        )
