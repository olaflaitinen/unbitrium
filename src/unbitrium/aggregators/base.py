"""
Base Aggregator class.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple

class Aggregator(ABC):
    """
    Abstract base class for server-side aggregation strategies.
    """

    @abstractmethod
    def aggregate(
        self,
        updates: List[Dict[str, Any]],
        current_global_model: Any
    ) -> Tuple[Any, Dict[str, float]]:
        """
        Aggregates client updates into a new global model.

        Parameters
        ----------
        updates : List[Dict[str, Any]]
            List of updates from clients. Each update dict must contain:
            - 'client_id': int
            - 'weights': Any (state_dict or similar)
            - 'num_samples': int
        current_global_model : Any
            The current global model (for reference or differencing).

        Returns
        -------
        Tuple[Any, Dict[str, float]]
            (new_global_model, aggregation_metrics)
        """
        pass
