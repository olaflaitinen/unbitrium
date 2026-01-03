"""
FedCM Aggregator.
"""

from typing import Any, Dict, List, Tuple
import torch
from unbitrium.aggregators.base import Aggregator

class FedCM(Aggregator):
    """
    Federated Client Momentum (FedCM).

    Tracks momentum V_k at the client side.
    However, if we are in a simulation where clients are stateless (sampled),
    momentum must be passed back and forth or stored in server state.

    Here we assume 'updates' contains the 'momentum' vector if computed locally,
    OR the aggregator computes it?

    Usually FedCM implies the *Algorithm* changes, not just aggregation.
    But as an aggregator, we might apply momentum to the update.

    Let's implement Server Momentum (FedAvgM) if Client Momentum is not possible purely in Aggregator.
    BUT spec says "Client-level momentum".
    This implies we need to store v_k.

    Assumption: Simulation Engine passes 'id' in updates.
    """

    def __init__(self, beta: float = 0.9):
        self.beta = beta
        self.velocities: Dict[str, Dict[str, torch.Tensor]] = {} # ClientID -> StateDict (V)

    def aggregate(
        self,
        updates: List[Dict[str, Any]],
        current_global_model: torch.nn.Module
    ) -> Tuple[torch.nn.Module, Dict[str, float]]:

        # In this implementation, we apply the momentum update step conceptually here
        # But ordinarily FedCM modifies the CLIENT update rule.
        # If 'updates' are just weights W_k, we can infer gradient G_k = W_global - W_k
        # And update V_k = beta * V_k + G_k
        # Then the real update to global is based on V_k?

        if not updates:
            return current_global_model, {}

        # 1. Infer Gradients and Update Momentum
        global_sd = current_global_model.state_dict()

        corrected_diffs = []
        total_samples = 0

        for u in updates:
            cid = str(u.get("client_id", "unknown")) # Requires ID to persist state
            w_k = u["state_dict"]
            num_samples = u["num_samples"]
            total_samples += num_samples

            # Init V_k
            if cid not in self.velocities:
                self.velocities[cid] = {k: torch.zeros_like(v) for k,v in global_sd.items()}

            # Compute Pseudo-Gradient (-Delta)
            # grad = w_global - w_k  (assuming lr=1 effectively, or provided)
            # For simplicity, let's treat update as the Step.

            diff = {}
            for k in global_sd:
                # Assuming simple SGD at client: w_k = w_g - lr * g
                # So (w_g - w_k) proportional to g
                delta = global_sd[k].float() - w_k[k].float()

                # Update Momentum
                # v_new = beta * v_old + delta
                self.velocities[cid][k] = self.beta * self.velocities[cid][k] + delta

                # The effective update used for aggregation is the Momentum Vector
                diff[k] = self.velocities[cid][k]

            corrected_diffs.append((diff, num_samples))

        # 2. Aggregate Velocities (Diffs)
        # W_new = W_old - Avg(Velocities) ??
        # Or usually FedCM aggregates 'Models' trained with partial momentum.
        # Given the ambiguity in "Aggregator" vs "Trainer", we'll do Standard FedAvg on "Momentum-Corrected Models"
        # pseudo-model = W_g - V_k

        if total_samples == 0:
            return current_global_model, {}

        avg_diff = {k: torch.zeros_like(v).float() for k,v in global_sd.items()}

        for diff, n in corrected_diffs:
            weight = n / total_samples
            for k in avg_diff:
                avg_diff[k] += diff[k] * weight

        # Apply to global
        new_sd = {}
        for k in global_sd:
            # W_new = W_old - AvgDiff
            new_sd[k] = global_sd[k] - avg_diff[k]
             # Restore dtype
            new_sd[k] = new_sd[k].to(global_sd[k].dtype)

        current_global_model.load_state_dict(new_sd)

        return current_global_model, {"fedcm_beta": self.beta}
