"""
AFL-DCS Aggregator.
"""

from typing import Any, Dict, List, Tuple
import torch
from unbitrium.aggregators.base import Aggregator

class AFL_DCS(Aggregator):
    r"""
    Asynchronous Federated Learning with Dynamic Client Scheduling (AFL-DCS).

    Handles asynchronous updates where 'updates' list might contain
    results from different timestamps (staleness).

    Actually, the Simulation Engine usually handles the Async Event Loop.
    The Aggregator here defines how to MERGE a stale update into the current model.

    Rule:
    W_new = (1 - alpha) * W_curr + alpha * W_stale
    where alpha usually depends on staleness function s(t - tau).
    """

    def __init__(self, alpha_base: float = 0.5, staleness_func="polynomial"):
        self.alpha_base = alpha_base
        self.staleness_func = staleness_func

    def aggregate(
        self,
        updates: List[Dict[str, Any]],
        current_global_model: torch.nn.Module
    ) -> Tuple[torch.nn.Module, Dict[str, float]]:

        # Async usually implies Sequential updates (one by one) or small batches.
        # Unbitrium Engine might send a batch of generic completed tasks.

        if not updates:
            return current_global_model, {}

        global_sd = current_global_model.state_dict()

        # We update incrementally for each update in the list
        # W <- (1-a)W + a W_k

        # We need access to 'staleness'. Assuming it's in update metadata
        # update["staleness"] = current_round - client_start_round

        for u in updates:
            staleness = u.get("staleness", 0)

            # Compute alpha(staleness)
            # Example: alpha / (staleness + 1)
            # Using simple constant decay for demo
            alpha = self.alpha_base / (1.0 + 0.5 * staleness)

            w_k = u["state_dict"]

            for k in global_sd:
                # Running average update
                # w_g = (1 - alpha) * w_g + alpha * w_k
                # w_g = w_g - alpha * (w_g - w_k)
                if isinstance(global_sd[k], torch.Tensor):
                    w_g = global_sd[k].float()
                    update_val = w_k[k].float()

                    new_val = (1 - alpha) * w_g + alpha * update_val
                    global_sd[k].copy_(new_val.to(global_sd[k].dtype))

        return current_global_model, {"processed_async_updates": len(updates)}
