"""
Deterministic Random Number Generation Management.
"""

import random
import numpy as np
import torch
from typing import Optional

class RNGManager:
    """
    Manages random seeds and generators for reproducibility.
    """

    def __init__(self, seed: int):
        self.seed = seed
        self.set_global_seeds(seed)

    def set_global_seeds(self, seed: int) -> None:
        """
        Sets the global random seeds for Python, NumPy, and PyTorch.

        Parameters
        ----------
        seed : int
            The seed value.
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
        """
        Returns a new NumPy random generator derived from the base seed.

        Parameters
        ----------
        delta : int
            Offset to add to the base seed to create a distinct sequence.

        Returns
        -------
        np.random.Generator
            Authenticated random generator.
        """
        return np.random.default_rng(self.seed + delta)
