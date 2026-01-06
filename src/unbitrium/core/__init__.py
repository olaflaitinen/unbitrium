"""Core package for Unbitrium.

Provides utilities, configuration, and base classes for federated learning.

Author: Olaf Yunus Laitinen Imanov <oyli@dtu.dk>
License: EUPL-1.2
"""

from __future__ import annotations

from unbitrium.core.utils import (
    get_provenance_info,
    set_global_seed,
    set_seed,
    setup_logging,
)

__all__ = [
    "setup_logging",
    "set_seed",
    "set_global_seed",
    "get_provenance_info",
]
