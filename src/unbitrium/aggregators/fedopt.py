"""
FedOpt Aggregators (FedAdam, FedYogi, FedAdagrad).
"""

from typing import Any, Dict, List, Tuple
from unbitrium.aggregators.base import Aggregator

class FedOpt(Aggregator):
    """
    Base class for server-side optimization aggregators.
    """
    def __init__(self, learning_rate: float = 0.01, beta1: float = 0.9, beta2: float = 0.999, tau: float = 1e-3):
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.tau = tau
        self.m = None
        self.v = None

    def aggregate(
        self,
        updates: List[Dict[str, Any]],
        current_global_model: Any
    ) -> Tuple[Any, Dict[str, float]]:
        # Compute pseudo-gradient Delta
        # Update m, v
        # Apply step
        return current_global_model, {}

class FedAdam(FedOpt):
    """
    FedAdam.
    """
    pass

class FedYogi(FedOpt):
    """
    FedYogi.
    """
    pass

class FedAdagrad(FedOpt):
    """
    FedAdagrad.
    """
    pass
