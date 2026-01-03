"""
pFedSim Aggregator.
"""

from typing import Any, Dict, List, Tuple
import torch

from unbitrium.aggregators.fedsim import FedSim

class pFedSim(FedSim):
    """
    Personalized FedSim (pFedSim).

    Aggregates only SHARED layers based on similarity.
    Personalized layers (Heads) are not aggregated or are handled separately.
    """

    def __init__(self, shared_layer_names: List[str] = None):
        """
        Args:
            shared_layer_names: List of keys (prefixes) to aggregate. If None, assumes all.
                                In real usage, this should be configured to exclude heads.
        """
        self.shared_layer_names = shared_layer_names or []

    def aggregate(
        self,
        updates: List[Dict[str, Any]],
        current_global_model: torch.nn.Module
    ) -> Tuple[torch.nn.Module, Dict[str, float]]:

        # Filter updates to only include shared layers
        # In pFedSim, the global model usually only maintains the body.
        # But here we assume updates contain everything, and we selectively update the global.

        # 1. Identify keys to aggregate
        ref_keys = list(current_global_model.state_dict().keys())
        keys_to_agg = []
        if not self.shared_layer_names:
            keys_to_agg = ref_keys
        else:
            for k in ref_keys:
                if any(start in k for start in self.shared_layer_names):
                    keys_to_agg.append(k)

        # 2. Extract Sub-State Dicts
        sub_updates = []
        for u in updates:
            sd = u["state_dict"]
            filtered = {k: sd[k] for k in keys_to_agg if k in sd}
            sub_updates.append({"state_dict": filtered})

        # 3. Create a temporary 'sub-model' to leverage FedSim logic?
        # Alternatively, just use internal helpers manually.

        if not sub_updates:
            return current_global_model, {}

        # Re-use flattening logic from FedSim but only on filtered keys
        # ... copying FedSim logic for brevity/customization

        # Flatten global relevant parts
        global_sd = current_global_model.state_dict()
        filtered_global = {k: global_sd[k] for k in keys_to_agg}
        global_vec = self._flatten(filtered_global)

        sims = []
        flat_updates = []
        for u in sub_updates:
            flat = self._flatten(u["state_dict"])
            flat_updates.append(flat)
            cos_sim = torch.nn.functional.cosine_similarity(global_vec.unsqueeze(0), flat.unsqueeze(0)).item()
            sims.append(max(0.0, cos_sim))

        total_sim = sum(sims)
        if total_sim < 1e-9:
             weights = [1.0 / len(updates)] * len(updates)
        else:
             weights = [s / total_sim for s in sims]

        new_flat = torch.zeros_like(global_vec)
        for w, flat in zip(weights, flat_updates):
            new_flat += w * flat

        new_sub_state = self._unflatten(new_flat, filtered_global)

        # Merge back into global
        final_sd = global_sd.copy()
        final_sd.update(new_sub_state)
        current_global_model.load_state_dict(final_sd)

        return current_global_model, {"avg_shared_similarity": float(sum(sims)/len(sims))}
