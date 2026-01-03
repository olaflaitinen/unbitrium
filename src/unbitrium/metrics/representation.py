"""
Representation metrics (NMI, CKA).
"""

import numpy as np
from sklearn.metrics import normalized_mutual_info_score

def compute_nmi(labels_true: np.ndarray, labels_pred: np.ndarray) -> float:
    """
    Computes Normalized Mutual Information.
    """
    return normalized_mutual_info_score(labels_true, labels_pred)

def compute_cka(K: np.ndarray, L: np.ndarray) -> float:
    """
    Computes Centered Kernel Alignment (CKA) between two kernels.
    """
    # HSIC logic placeholder
    return 0.0
