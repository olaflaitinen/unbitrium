"""System metrics for federated learning.

Provides computation and communication efficiency metrics.

Author: Olaf Yunus Laitinen Imanov <oyli@dtu.dk>
License: EUPL-1.2
"""

from __future__ import annotations


import numpy as np
import torch


def compute_model_size(
    model: torch.nn.Module | dict[str, torch.Tensor],
) -> dict[str, int]:
    """Compute model size metrics.

    Args:
        model: PyTorch model or state dictionary.

    Returns:
        Dictionary with 'num_parameters' and 'size_bytes'.
    """
    if isinstance(model, torch.nn.Module):
        state_dict = model.state_dict()
    else:
        state_dict = model

    num_params = 0
    size_bytes = 0

    for value in state_dict.values():
        if isinstance(value, torch.Tensor):
            num_params += value.numel()
            size_bytes += value.element_size() * value.numel()

    return {
        "num_parameters": num_params,
        "size_bytes": size_bytes,
    }


def compute_flops_estimate(
    model: torch.nn.Module,
    input_shape: tuple[int, ...],
) -> int:
    """Estimate FLOPs for a forward pass.

    Args:
        model: PyTorch model.
        input_shape: Input tensor shape (excluding batch).

    Returns:
        Estimated FLOPs.
    """
    total_flops = 0

    for module in model.modules():
        if isinstance(module, torch.nn.Linear):
            # FLOPs for linear: 2 * in_features * out_features
            total_flops += 2 * module.in_features * module.out_features
        elif isinstance(module, torch.nn.Conv2d):
            # FLOPs for conv: 2 * K^2 * C_in * C_out * H_out * W_out
            # Simplified estimate
            total_flops += 2 * (
                module.kernel_size[0]
                * module.kernel_size[1]
                * module.in_channels
                * module.out_channels
            )

    return total_flops


def compute_throughput(
    num_samples: int,
    elapsed_time: float,
) -> float:
    """Compute training throughput.

    Args:
        num_samples: Number of samples processed.
        elapsed_time: Time in seconds.

    Returns:
        Samples per second.
    """
    if elapsed_time <= 0:
        return 0.0
    return num_samples / elapsed_time


def compute_round_statistics(
    round_times: list[float],
) -> dict[str, float]:
    """Compute statistics over round completion times.

    Args:
        round_times: List of round completion times.

    Returns:
        Dictionary with mean, std, min, max times.
    """
    if not round_times:
        return {
            "mean_time": 0.0,
            "std_time": 0.0,
            "min_time": 0.0,
            "max_time": 0.0,
        }

    times = np.array(round_times)
    return {
        "mean_time": float(np.mean(times)),
        "std_time": float(np.std(times)),
        "min_time": float(np.min(times)),
        "max_time": float(np.max(times)),
    }
