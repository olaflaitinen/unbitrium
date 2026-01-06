"""Deterministic Random Number Generation for Reproducible FL.

Provides centralized management of random seeds across Python, NumPy,
and PyTorch to ensure reproducible federated learning experiments.

Author: Olaf Yunus Laitinen Imanov <oyli@dtu.dk>
License: EUPL-1.2
"""

from __future__ import annotations

import random

import numpy as np
import torch


class RNGManager:
    """Manages random seeds and generators for reproducibility.

    Provides a centralized mechanism for setting global seeds and
    creating independent local generators for different simulation
    components.

    Args:
        seed: Base seed for all random number generation.

    Example:
        >>> rng_manager = RNGManager(seed=42)
        >>> local_rng = rng_manager.get_local_rng(round_num)
    """

    def __init__(self, seed: int) -> None:
        """Initialize RNG manager with base seed.

        Args:
            seed: Base random seed.
        """
        self.seed = seed
        self.set_global_seeds(seed)

    def set_global_seeds(self, seed: int) -> None:
        """Set global random seeds for all libraries.

        Sets seeds for Python's random, NumPy, and PyTorch
        (including CUDA if available).

        Args:
            seed: The seed value to set globally.
        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def get_local_rng(self, delta: int = 0) -> np.random.Generator:
        """Get a local random generator with derived seed.

        Creates a new NumPy random generator with a seed derived
        from the base seed plus an offset. Useful for creating
        independent random streams for different rounds or clients.

        Args:
            delta: Offset to add to base seed for unique sequence.

        Returns:
            NumPy random Generator instance.

        Example:
            >>> round_rng = rng_manager.get_local_rng(round_num)
            >>> client_ids = round_rng.choice(100, size=10)
        """
        return np.random.default_rng(self.seed + delta)

    def get_torch_generator(self, delta: int = 0) -> torch.Generator:
        """Get a PyTorch generator with derived seed.

        Args:
            delta: Offset to add to base seed.

        Returns:
            PyTorch Generator instance for use in DataLoaders.
        """
        gen = torch.Generator()
        gen.manual_seed(self.seed + delta)
        return gen

    def fork(self, delta: int) -> "RNGManager":
        """Create a child RNGManager with a derived seed.

        Args:
            delta: Offset to add to base seed.

        Returns:
            New RNGManager instance with derived seed.
        """
        return RNGManager(self.seed + delta)
