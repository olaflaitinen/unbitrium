"""Privacy metrics for federated learning.

Provides functions to estimate privacy leakage and differential privacy properties.

Author: Olaf Yunus Laitinen Imanov <oyli@dtu.dk>
License: EUPL-1.2
"""

from __future__ import annotations


import numpy as np
import torch


def compute_privacy_metrics(
    model_updates: list[dict[str, torch.Tensor]],
    epsilon: float | None = None,
    delta: float = 1e-5,
) -> dict[str, float]:
    """Compute privacy-related metrics.

    Args:
        model_updates: List of client model state dictionaries.
        epsilon: Optional differential privacy epsilon.
        delta: Differential privacy delta.

    Returns:
        Dictionary of privacy metrics.
    """
    if not model_updates:
        return {
            "avg_update_norm": 0.0,
            "max_update_norm": 0.0,
            "update_similarity": 0.0,
        }

    # Compute L2 norms of updates
    norms = []
    flattened = []
    for update in model_updates:
        tensors = []
        for value in update.values():
            if isinstance(value, torch.Tensor):
                tensors.append(value.view(-1).float())
        if tensors:
            flat = torch.cat(tensors)
            flattened.append(flat)
            norms.append(torch.norm(flat).item())

    avg_norm = float(np.mean(norms)) if norms else 0.0
    max_norm = float(np.max(norms)) if norms else 0.0

    # Pairwise similarity
    similarities = []
    for i in range(len(flattened)):
        for j in range(i + 1, len(flattened)):
            sim = torch.nn.functional.cosine_similarity(
                flattened[i].unsqueeze(0),
                flattened[j].unsqueeze(0),
            ).item()
            similarities.append(sim)

    avg_similarity = float(np.mean(similarities)) if similarities else 0.0

    result = {
        "avg_update_norm": avg_norm,
        "max_update_norm": max_norm,
        "update_similarity": avg_similarity,
        "num_updates": float(len(model_updates)),
    }

    if epsilon is not None:
        result["epsilon"] = epsilon
        result["delta"] = delta
        # Approximate privacy amplification from subsampling
        result["effective_epsilon"] = epsilon * np.sqrt(2 * np.log(1.25 / delta))

    return result


def compute_gradient_sensitivity(
    gradients: torch.Tensor,
    clip_norm: float | None = None,
) -> dict[str, float]:
    """Compute gradient sensitivity metrics.

    Args:
        gradients: Tensor of gradients.
        clip_norm: Optional gradient clipping norm.

    Returns:
        Dictionary of sensitivity metrics.
    """
    grad_norm = torch.norm(gradients).item()

    result = {
        "gradient_norm": grad_norm,
    }

    if clip_norm is not None:
        result["clip_norm"] = clip_norm
        result["clipped"] = float(grad_norm > clip_norm)
        result["effective_sensitivity"] = min(grad_norm, clip_norm)
    else:
        result["effective_sensitivity"] = grad_norm

    return result
