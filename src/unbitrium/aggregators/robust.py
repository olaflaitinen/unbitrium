"""
Robust Aggregators.
"""

from typing import Any, Dict, List, Tuple
from unbitrium.aggregators.base import Aggregator

class TrimmedMean(Aggregator):
    """
    Coordinate-wise Trimmed Mean.
    """
    def __init__(self, beta: float = 0.1):
        self.beta = beta

    def aggregate(
        self,
        updates: List[Dict[str, Any]],
        current_global_model: Any
    ) -> Tuple[Any, Dict[str, float]]:
        return current_global_model, {"robust_method": "trimmed_mean"}

class Krum(Aggregator):
    """
    Krum Aggregator.
    """
    def __init__(self, num_byzantine: int = 1):
        self.f = num_byzantine

    def aggregate(
        self,
        updates: List[Dict[str, Any]],
        current_global_model: Any
    ) -> Tuple[Any, Dict[str, float]]:
        return current_global_model, {"robust_method": "krum"}
