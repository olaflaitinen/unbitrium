"""Optimization metrics for federated learning.

Provides convergence and optimization-related metrics.

Author: Olaf Yunus Laitinen Imanov <oyli@dtu.dk>
License: EUPL-1.2
"""

from __future__ import annotations

import numpy as np


def compute_convergence_rate(
    loss_history: list[float],
    window_size: int = 5,
) -> float:
    """Compute convergence rate from loss history.

    Args:
        loss_history: List of loss values per round.
        window_size: Window size for smoothing.

    Returns:
        Average rate of loss decrease.
    """
    if len(loss_history) < 2:
        return 0.0

    losses = np.array(loss_history)

    # Compute rolling average
    if len(losses) > window_size:
        smoothed = np.convolve(losses, np.ones(window_size) / window_size, mode="valid")
    else:
        smoothed = losses

    # Compute rate of decrease
    if len(smoothed) < 2:
        return 0.0

    rates = []
    for i in range(1, len(smoothed)):
        if smoothed[i - 1] > 0:
            rate = (smoothed[i - 1] - smoothed[i]) / smoothed[i - 1]
            rates.append(rate)

    return float(np.mean(rates)) if rates else 0.0


def compute_rounds_to_accuracy(
    accuracy_history: list[float],
    target_accuracy: float,
) -> int:
    """Compute number of rounds to reach target accuracy.

    Args:
        accuracy_history: List of accuracy values per round.
        target_accuracy: Target accuracy to reach.

    Returns:
        Number of rounds, or -1 if not reached.
    """
    for i, acc in enumerate(accuracy_history):
        if acc >= target_accuracy:
            return i + 1
    return -1


def compute_communication_efficiency(
    accuracy_history: list[float],
    bytes_per_round: int,
) -> float:
    """Compute communication efficiency (accuracy per byte).

    Args:
        accuracy_history: List of accuracy values.
        bytes_per_round: Bytes communicated per round.

    Returns:
        Final accuracy divided by total bytes.
    """
    if not accuracy_history or bytes_per_round <= 0:
        return 0.0

    total_bytes = len(accuracy_history) * bytes_per_round
    return accuracy_history[-1] / total_bytes
