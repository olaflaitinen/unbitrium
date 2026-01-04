"""Heterogeneity metrics for federated learning.

Provides functions to quantify data and model heterogeneity across
clients in federated learning settings.

Author: Olaf Yunus Laitinen Imanov <oyli@dtu.dk>
License: EUPL-1.2
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch


def compute_gradient_variance(
    local_models: list[dict[str, torch.Tensor]],
    global_model: dict[str, torch.Tensor],
) -> float:
    """Compute variance of local model weights relative to global model.

    Mathematical formulation:

    $$
    \\sigma^2 = \\frac{1}{K} \\sum_{k=1}^K \\| w_k - w_g \\|^2
    $$

    Args:
        local_models: List of client model state dictionaries.
        global_model: Global model state dictionary.

    Returns:
        Average squared L2 distance from global model.
    """
    if not local_models:
        return 0.0

    variance = 0.0
    for w_k in local_models:
        diff_norm_sq = 0.0
        for key in global_model:
            if key in w_k and isinstance(global_model[key], torch.Tensor):
                diff = w_k[key].float() - global_model[key].float()
                diff_norm_sq += torch.sum(diff**2).item()
        variance += diff_norm_sq

    return variance / len(local_models)


def compute_drift_norm(
    initial_model: dict[str, torch.Tensor],
    final_model: dict[str, torch.Tensor],
) -> float:
    """Compute L2 norm of weight drift between two model states.

    Args:
        initial_model: Initial model state dictionary.
        final_model: Final model state dictionary.

    Returns:
        L2 norm of the weight difference.
    """
    norm_sq = 0.0
    for key in initial_model:
        if isinstance(initial_model[key], torch.Tensor):
            diff = final_model[key].float() - initial_model[key].float()
            norm_sq += torch.sum(diff**2).item()
    return float(np.sqrt(norm_sq))


def compute_imbalance_ratio(client_sample_counts: list[int]) -> float:
    """Compute imbalance ratio of client dataset sizes.

    Args:
        client_sample_counts: List of sample counts per client.

    Returns:
        Ratio of maximum to minimum sample count.
    """
    if not client_sample_counts:
        return 0.0

    max_samples = max(client_sample_counts)
    min_samples = min(client_sample_counts)

    if min_samples == 0:
        return float("inf")
    return max_samples / min_samples


def compute_label_entropy(
    labels: np.ndarray,
    client_indices: dict[int, list[int]],
) -> float:
    """Compute average label entropy across clients.

    Higher entropy indicates more balanced label distributions;
    lower entropy indicates more skewed distributions.

    Args:
        labels: Array of class labels.
        client_indices: Mapping from client ID to sample indices.

    Returns:
        Average entropy across all clients.
    """
    num_classes = len(np.unique(labels))
    entropies = []

    for indices in client_indices.values():
        if len(indices) == 0:
            continue

        client_labels = labels[indices]
        counts = np.bincount(client_labels, minlength=num_classes)
        probs = counts / counts.sum()
        probs = probs[probs > 0]  # Avoid log(0)
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        entropies.append(entropy)

    return float(np.mean(entropies)) if entropies else 0.0


def compute_emd(
    labels: np.ndarray,
    client_indices: dict[int, list[int]],
) -> float:
    """Compute average Earth Mover's Distance from global distribution.

    Args:
        labels: Array of class labels.
        client_indices: Mapping from client ID to sample indices.

    Returns:
        Average EMD across all clients.
    """
    num_classes = len(np.unique(labels))

    # Global distribution
    global_counts = np.bincount(labels, minlength=num_classes)
    global_dist = global_counts / global_counts.sum()

    emds = []
    for indices in client_indices.values():
        if len(indices) == 0:
            continue

        client_labels = labels[indices]
        client_counts = np.bincount(client_labels, minlength=num_classes)
        client_dist = client_counts / client_counts.sum()

        # L1 distance as simplified EMD for 1D distributions
        emd = np.sum(np.abs(client_dist - global_dist))
        emds.append(emd)

    return float(np.mean(emds)) if emds else 0.0


def compute_js_divergence(
    labels: np.ndarray,
    client_indices: dict[int, list[int]],
) -> float:
    """Compute average Jensen-Shannon divergence from global distribution.

    Args:
        labels: Array of class labels.
        client_indices: Mapping from client ID to sample indices.

    Returns:
        Average JS divergence across all clients.
    """
    num_classes = len(np.unique(labels))

    # Global distribution
    global_counts = np.bincount(labels, minlength=num_classes)
    global_dist = global_counts / global_counts.sum()

    divergences = []
    for indices in client_indices.values():
        if len(indices) == 0:
            continue

        client_labels = labels[indices]
        client_counts = np.bincount(client_labels, minlength=num_classes)
        client_dist = client_counts / client_counts.sum()

        # Jensen-Shannon divergence
        m = 0.5 * (client_dist + global_dist)
        eps = 1e-10

        kl_p_m = np.sum(client_dist * np.log((client_dist + eps) / (m + eps)))
        kl_q_m = np.sum(global_dist * np.log((global_dist + eps) / (m + eps)))
        js = 0.5 * (kl_p_m + kl_q_m)

        divergences.append(js)

    return float(np.mean(divergences)) if divergences else 0.0


def compute_nmi(
    partition_indices: dict[int, list[int]],
    targets: np.ndarray,
) -> float:
    """Compute Normalized Mutual Information between partitions and labels.

    Args:
        partition_indices: Mapping from client ID to sample indices.
        targets: Array of class labels.

    Returns:
        NMI score, or -1.0 if sklearn is not available.
    """
    try:
        from sklearn.metrics import normalized_mutual_info_score
    except ImportError:
        return -1.0

    client_ids = []
    class_labels = []

    for cid, indices in partition_indices.items():
        for idx in indices:
            client_ids.append(cid)
            class_labels.append(targets[idx])

    return float(normalized_mutual_info_score(class_labels, client_ids))
