"""Datasets package for Unbitrium.

Provides dataset utilities and registry for federated learning experiments.

Author: Olaf Yunus Laitinen Imanov <oyli@dtu.dk>
License: EUPL-1.2
"""

from __future__ import annotations

from unbitrium.datasets.registry import DatasetRegistry, register_dataset, get_dataset

__all__ = [
    "DatasetRegistry",
    "register_dataset",
    "get_dataset",
]
