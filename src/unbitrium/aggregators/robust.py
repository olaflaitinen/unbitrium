"""Robust aggregation utilities.

Provides base utilities for Byzantine-robust aggregation.
Note: Primary implementations are in krum.py and trimmed_mean.py.

Author: Olaf Yunus Laitinen Imanov <oyli@dtu.dk>
License: EUPL-1.2
"""

from __future__ import annotations

from typing import Any

import torch
import numpy as np


def compute_pairwise_distances(
    updates: list[dict[str, torch.Tensor]],
) -> np.ndarray:
    """Compute pairwise L2 distances between client updates.

    Args:
        updates: List of client state dictionaries.

    Returns:
        Pairwise distance matrix.
    """
    num_clients = len(updates)

    # Flatten each update
    flattened = []
    for update in updates:
        tensors = []
        for value in update.values():
            if isinstance(value, torch.Tensor):
                tensors.append(value.view(-1).float())
        if tensors:
            flattened.append(torch.cat(tensors))
        else:
            flattened.append(torch.tensor([]))

    # Compute distances
    distances = np.zeros((num_clients, num_clients))
    for i in range(num_clients):
        for j in range(i + 1, num_clients):
            if flattened[i].numel() > 0 and flattened[j].numel() > 0:
                dist = torch.norm(flattened[i] - flattened[j]).item()
                distances[i, j] = dist
                distances[j, i] = dist

    return distances


def coordinate_wise_median(
    updates: list[dict[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    """Compute coordinate-wise median of updates.

    Args:
        updates: List of client state dictionaries.

    Returns:
        Median state dictionary.
    """
    if not updates:
        return {}

    result: dict[str, torch.Tensor] = {}
    first_state = updates[0]

    for key in first_state.keys():
        if isinstance(first_state[key], torch.Tensor):
            stacked = torch.stack([u[key].float() for u in updates], dim=0)
            median_val, _ = torch.median(stacked, dim=0)
            result[key] = median_val.to(first_state[key].dtype)
        else:
            result[key] = first_state[key]

    return result


def detect_outliers(
    updates: list[dict[str, torch.Tensor]],
    threshold: float = 2.0,
) -> list[int]:
    """Detect outlier updates based on L2 norm.

    Args:
        updates: List of client state dictionaries.
        threshold: Standard deviation threshold for outliers.

    Returns:
        List of outlier indices.
    """
    norms = []
    for update in updates:
        total_norm = 0.0
        for value in update.values():
            if isinstance(value, torch.Tensor):
                total_norm += torch.sum(value.float() ** 2).item()
        norms.append(np.sqrt(total_norm))

    norms = np.array(norms)
    mean_norm = np.mean(norms)
    std_norm = np.std(norms)

    if std_norm < 1e-10:
        return []

    outliers = []
    for i, norm in enumerate(norms):
        if abs(norm - mean_norm) > threshold * std_norm:
            outliers.append(i)

    return outliers
