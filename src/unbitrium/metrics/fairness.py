"""
Fairness Metrics.
"""

from typing import List, Dict
import numpy as np

def compute_jains_index(values: List[float]) -> float:
    """
    Computes Jain's Fairness Index.

    J = (\sum x_i)^2 / (n * \sum x_i^2)

    Used for accuracy fairness, selection fairness, etc.
    """
    if not values:
        return 0.0

    vals = np.array(values)
    n = len(vals)
    sum_sq = np.sum(vals) ** 2
    sq_sum = np.sum(vals ** 2)

    if sq_sum == 0:
        return 1.0 # All zeros is "fair"? Or 0? Usually handled as corner case.

    return float(sum_sq / (n * sq_sum))

def compute_selection_bias(
    selected_counts: Dict[int, int],
    total_rounds: int,
    num_subsampled: int
) -> float:
    """
    Computes a metric for selection bias.
    Simple Metric: StdDev of Selection Counts normalized by Expected Count.
    """
    if not selected_counts:
        return 0.0

    counts = np.array(list(selected_counts.values()))

    # Expected count if uniform random
    # p = num_subsampled / num_clients (per round)
    # expected = total_rounds * p (But we need N_clients)

    # We just return CV (Coefficient of Variation) of counts
    mean = np.mean(counts)
    if mean == 0:
        return 0.0
    std = np.std(counts)

    return float(std / mean)
