"""
Distributional metrics (EMD, JS, Entropy).
"""

import numpy as np
from scipy.spatial.distance import jensenshannon
from scipy.stats import entropy
try:
    import ot
except ImportError:
    ot = None

def compute_emd(p: np.ndarray, q: np.ndarray) -> float:
    """
    Computes Earth Mover's Distance between two probability distributions.

    Parameters
    ----------
    p : np.ndarray
        First distribution (1D).
    q : np.ndarray
        Second distribution (1D).

    Returns
    -------
    float
        EMD value.
    """
    if ot is not None:
        # Use simple 1D Wasserstein if applicable, or OT for generic
        # For 1D histograms with assumed metric ground distance (e.g. class indices),
        # we can use:
        M = ot.dist(np.arange(len(p)).reshape(-1,1), np.arange(len(q)).reshape(-1,1))
        return float(ot.emd2(p, q, M))
    else:
        # Simple L1 accumulation for 1D case as fallback
        return float(np.sum(np.abs(np.cumsum(p) - np.cumsum(q))))

def compute_js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """
    Computes Jensen-Shannon Divergence.
    """
    return float(jensenshannon(p, q) ** 2)

def compute_label_entropy(p: np.ndarray) -> float:
    """
    Computes Shannon entropy of the label distribution.
    """
    return float(entropy(p))
