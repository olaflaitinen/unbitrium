"""
FedDyn Aggregator.
"""

from typing import Any, Dict, List, Tuple
import torch

from unbitrium.aggregators.base import Aggregator

class FedDyn(Aggregator):
    """
    Federated Dynamic Regularization (FedDyn).

    Paper: "Federated Learning on Non-IID Data via Dynamic Regularization" (ICLR 2021).

    Server maintains state `h_server`.
    Aggregation rule:
    $$
    w^{t+1} = \frac{1}{N} \sum_k w_k^{t+1} - \frac{1}{\alpha} h^{t+1}
    $$
    Updates h:
    $$
    h^{t+1} = h^t - \alpha (w^{t+1} - \text{avg}(w_k))
    $$
    """

    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha
        self.h_server: Dict[str, torch.Tensor] = {}

    def aggregate(
        self,
        updates: List[Dict[str, Any]],
        current_global_model: torch.nn.Module
    ) -> Tuple[torch.nn.Module, Dict[str, float]]:

        if not updates:
            return current_global_model, {}

        num_clients = len(updates) # FedDyn usually assumes equal weighting or treats clients as tasks
        # Assuming equal weight sum first for simplicity of Formula

        # 1. Compute Average of client Models
        avg_state_dict = {}
        first_state = updates[0]["state_dict"]

        for key in first_state.keys():
            if isinstance(first_state[key], torch.Tensor):
                avg_val = torch.zeros_like(first_state[key], dtype=torch.float32) # Accumulate in float32
                for u in updates:
                    avg_val += u["state_dict"][key].to(torch.float32)
                avg_val /= num_clients
                avg_state_dict[key] = avg_val

        # 2. Update Server State h
        # Initialize h if empty
        if not self.h_server:
             for k, v in avg_state_dict.items():
                 self.h_server[k] = torch.zeros_like(v)

        # 3. Compute New Global Model
        # w_new = avg_model - (1/alpha) * h_old
        new_state_dict = {}
        for key, avg_w in avg_state_dict.items():
             h = self.h_server[key]
             new_w = avg_w - (1.0 / self.alpha) * h
             new_state_dict[key] = new_w

        # 4. Update h_server for next round
        # h_new = h_old - alpha * (w_new - avg_w)
        for key in avg_state_dict.keys():
            self.h_server[key] -= self.alpha * (new_state_dict[key] - avg_state_dict[key])

        # Load back
        # Note: FedDyn weights might perform outside valid range if strict bounds exist (e.g. BatchNorm), careful.
        # Converting back to original dtype
        final_state_dict = {}
        original_state = current_global_model.state_dict()
        for k, v in new_state_dict.items():
            final_state_dict[k] = v.to(original_state[k].dtype)

        current_global_model.load_state_dict(final_state_dict)

        return current_global_model, {"feddyn_alpha": self.alpha}
