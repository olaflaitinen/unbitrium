"""Privacy package for Unbitrium.

Provides differential privacy mechanisms for federated learning.

Author: Olaf Yunus Laitinen Imanov <oyli@dtu.dk>
License: EUPL-1.2
"""

from __future__ import annotations

from unbitrium.privacy.mechanism import (
    GaussianMechanism,
    LaplaceMechanism,
    clip_gradients,
)

__all__ = [
    "GaussianMechanism",
    "LaplaceMechanism",
    "clip_gradients",
]
