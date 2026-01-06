"""Fairness metrics for federated learning.

Provides functions to measure fairness across clients.

Author: Olaf Yunus Laitinen Imanov <oyli@dtu.dk>
License: EUPL-1.2
"""

from __future__ import annotations


import numpy as np


def compute_fairness_metrics(
    accuracies: list[float] | np.ndarray,
) -> dict[str, float]:
    """Compute fairness metrics across client accuracies.

    Args:
        accuracies: List of per-client accuracies.

    Returns:
        Dictionary of fairness metrics.
    """
    accs = np.asarray(accuracies)

    if len(accs) == 0:
        return {
            "min_accuracy": 0.0,
            "max_accuracy": 0.0,
            "avg_accuracy": 0.0,
            "std_accuracy": 0.0,
            "worst_10p_accuracy": 0.0,
            "accuracy_gap": 0.0,
            "jains_fairness_index": 0.0,
        }

    # Basic statistics
    min_acc = float(np.min(accs))
    max_acc = float(np.max(accs))
    avg_acc = float(np.mean(accs))
    std_acc = float(np.std(accs))

    # Worst 10% accuracy
    k = max(1, len(accs) // 10)
    worst_10p = float(np.mean(np.sort(accs)[:k]))

    # Accuracy gap (max - min)
    gap = max_acc - min_acc

    # Jain's fairness index
    # J(x) = (sum(x))^2 / (n * sum(x^2))
    n = len(accs)
    jains = (np.sum(accs) ** 2) / (n * np.sum(accs**2)) if np.sum(accs**2) > 0 else 0.0

    return {
        "min_accuracy": min_acc,
        "max_accuracy": max_acc,
        "avg_accuracy": avg_acc,
        "std_accuracy": std_acc,
        "worst_10p_accuracy": worst_10p,
        "accuracy_gap": gap,
        "jains_fairness_index": float(jains),
    }


def compute_demographic_parity(
    predictions: np.ndarray,
    sensitive_attributes: np.ndarray,
) -> float:
    """Compute demographic parity difference.

    Args:
        predictions: Binary predictions.
        sensitive_attributes: Binary sensitive attribute.

    Returns:
        Demographic parity difference.
    """
    group_0 = predictions[sensitive_attributes == 0]
    group_1 = predictions[sensitive_attributes == 1]

    if len(group_0) == 0 or len(group_1) == 0:
        return 0.0

    rate_0 = np.mean(group_0)
    rate_1 = np.mean(group_1)

    return float(abs(rate_0 - rate_1))


def compute_equalized_odds_difference(
    predictions: np.ndarray,
    labels: np.ndarray,
    sensitive_attributes: np.ndarray,
) -> float:
    """Compute equalized odds difference.

    Args:
        predictions: Binary predictions.
        labels: True binary labels.
        sensitive_attributes: Binary sensitive attribute.

    Returns:
        Equalized odds difference.
    """
    diffs = []

    for y in [0, 1]:
        mask = labels == y
        preds = predictions[mask]
        sens = sensitive_attributes[mask]

        group_0 = preds[sens == 0]
        group_1 = preds[sens == 1]

        if len(group_0) > 0 and len(group_1) > 0:
            diff = abs(np.mean(group_0) - np.mean(group_1))
            diffs.append(diff)

    return float(np.mean(diffs)) if diffs else 0.0
