"""
FedAdam Aggregator.
"""

from typing import Any, Dict, List, Tuple
import torch
from unbitrium.aggregators.base import Aggregator

class FedAdam(Aggregator):
    """
    FedAdam (Server-side Adaptive Optimization).

    Treats the averaged client update as a pseudo-gradient ($\Delta$)
    and applies Adam update rule to the server model.
    """

    def __init__(self, lr: float = 0.01, beta1: float = 0.9, beta2: float = 0.99, epsilon: float = 1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

        self.m: Dict[str, torch.Tensor] = {} # First moment
        self.v: Dict[str, torch.Tensor] = {} # Second moment
        self.t = 0

    def aggregate(
        self,
        updates: List[Dict[str, Any]],
        current_global_model: torch.nn.Module
    ) -> Tuple[torch.nn.Module, Dict[str, float]]:

        if not updates:
            return current_global_model, {}

        # 1. Compute Pseudo-Gradient Delta = W_old - Avg(W_new)
        # Note: Reddi et al. define Delta_t = x_t - \sum w_k x_k
        # i.e. The direction to move (Gradient direction)

        total_samples = sum(u["num_samples"] for u in updates)
        metrics = {"total_samples": float(total_samples)}

        if total_samples == 0:
             return current_global_model, metrics

        global_sd = current_global_model.state_dict()

        # Avg Client Model
        avg_sd = {}
        first_state = updates[0]["state_dict"]
        for k in first_state.keys():
            if isinstance(first_state[k], torch.Tensor):
                weighted_sum = torch.zeros_like(first_state[k], dtype=torch.float32)
                for u in updates:
                     weighted_sum += u["state_dict"][k].float() * (u["num_samples"] / total_samples)
                avg_sd[k] = weighted_sum
            else:
                avg_sd[k] = first_state[k] # Meta param

        # 2. Update Step
        self.t += 1

        # Init state
        if not self.m:
            for k, v in avg_sd.items():
                if isinstance(v, torch.Tensor):
                    self.m[k] = torch.zeros_like(v)
                    self.v[k] = torch.zeros_like(v)

        new_sd = {}
        for k in avg_sd:
             if k in self.m:
                 # Delta = w_t - avg_w
                 delta = global_sd[k].float() - avg_sd[k]

                 # Adam Logic
                 # m_t = b1*m_{t-1} + (1-b1)*delta
                 self.m[k] = self.beta1 * self.m[k] + (1 - self.beta1) * delta

                 # v_t = b2*v_{t-1} + (1-b2)*delta^2
                 self.v[k] = self.beta2 * self.v[k] + (1 - self.beta2) * (delta ** 2)

                 # Bias correction? Usually omitted in FedOpt papers for simplicity or included.
                 # Let's include standard Adam bias correction
                 m_hat = self.m[k] / (1 - self.beta1 ** self.t)
                 v_hat = self.v[k] / (1 - self.beta2 ** self.t)

                 # Update
                 # w_{t+1} = w_t - lr * m_hat / (sqrt(v_hat) + eps)
                 step = self.lr * m_hat / (torch.sqrt(v_hat) + self.epsilon)
                 new_w = global_sd[k].float() - step

                 new_sd[k] = new_w.to(global_sd[k].dtype)
             else:
                 new_sd[k] = avg_sd[k]

        current_global_model.load_state_dict(new_sd)
        return current_global_model, metrics
