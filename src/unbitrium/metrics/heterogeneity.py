"""
Heterogeneity Metrics.
"""

from typing import List, Dict, Union
import torch
import numpy as np

def compute_gradient_variance(
    local_models: List[Dict[str, torch.Tensor]],
    global_model: Dict[str, torch.Tensor]
) -> float:
    """
    Computes the variance of local gradients (or updates) relative to the global model.

    Formula:
    Var = \sum || w_k - w_g ||^2 / K
    """
    if not local_models:
        return 0.0

    variance = 0.0
    for w_k in local_models:
        # Assuming w_k is state_dict
        # Compute L2 norm of difference for all params
        diff_norm_sq = 0.0
        for k in global_model:
            if k in w_k and isinstance(global_model[k], torch.Tensor):
                diff = w_k[k].float() - global_model[k].float()
                diff_norm_sq += torch.sum(diff ** 2).item()
        variance += diff_norm_sq

    return variance / len(local_models)

def compute_drift_norm(
    initial_model: Dict[str, torch.Tensor],
    final_model: Dict[str, torch.Tensor]
) -> float:
    """
    Computes the L2 norm of the weight drift (Update Magnitude).
    || w_t+1 - w_t ||
    """
    norm_sq = 0.0
    for k in initial_model:
        if isinstance(initial_model[k], torch.Tensor):
            diff = final_model[k].float() - initial_model[k].float()
            norm_sq += torch.sum(diff ** 2).item()
    return np.sqrt(norm_sq)

def compute_imbalance_ratio(client_sample_counts: List[int]) -> float:
    """
    Computes the Imbalance Ratio of the dataset partition sizes.
    IR = Max_Samples / Min_Samples
    """
    if not client_sample_counts:
        return 0.0

    mx = max(client_sample_counts)
    mn = min(client_sample_counts)

    if mn == 0:
        return float('inf')
    return mx / mn

def compute_nmi(partition_indices: Dict[int, List[int]], targets: np.ndarray) -> float:
    """
    Computes Normalized Mutual Information (NMI) between the partition assignments
    and the ground truth class labels.

    This measures how much 'class info' is contained in the 'client id'.
    If each client has only 1 class, NMI is high.
    If each client has uniform distribution, NMI is low.
    """
    try:
        from sklearn.metrics import normalized_mutual_info_score
    except ImportError:
        return -1.0

    # Construct "Predicted Cluster" (Client ID) and "True Class" arrays
    client_ids = []
    class_labels = []

    for cid, indices in partition_indices.items():
         for idx in indices:
             client_ids.append(cid)
             class_labels.append(targets[idx])

    return normalized_mutual_info_score(class_labels, client_ids)
