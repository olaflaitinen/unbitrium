"""
Optimization metrics (Gradient Variance, Drift).
"""

from typing import List, Any
import numpy as np

def compute_gradient_variance(gradients: List[np.ndarray]) -> float:
    """
    Computes the variance of client gradients.
    """
    # Assuming gradients is a list of flattened numpy arrays
    stacked = np.stack(gradients)
    mean_grad = np.mean(stacked, axis=0)
    variance = np.mean(np.linalg.norm(stacked - mean_grad, axis=1)**2)
    return float(variance)

def compute_drift_norm(w_local: Any, w_global: Any) -> float:
    """
    Computes Euclidean distance between local and global model.
    """
    # Placeholder for model walker
    return 0.0
