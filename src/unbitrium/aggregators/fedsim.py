"""
FedSim Aggregator.
"""

from typing import Any, Dict, List, Tuple
import torch
import torch.nn.functional as F
from unbitrium.aggregators.base import Aggregator

class FedSim(Aggregator):
    """
    Similarity-Guided Aggregator (FedSim).

    Weights client updates based on Cosine Similarity to the previous global model vector.
    """

    def aggregate(
        self,
        updates: List[Dict[str, Any]],
        current_global_model: torch.nn.Module
    ) -> Tuple[torch.nn.Module, Dict[str, float]]:

        if not updates:
            return current_global_model, {}

        # Flatten global model
        global_vec = self._flatten(current_global_model.state_dict())

        # Compute Similarities
        sims = []
        flat_updates = []
        for u in updates:
            flat = self._flatten(u["state_dict"])
            flat_updates.append(flat)
            # Cosine Sim
            # If flat is exactly global (no training), sim is 1.0
            # Usually we compare Update Vector (delta) or Model Vector?
            # FedSim usually compares Model Vectors.
            cos_sim = F.cosine_similarity(global_vec.unsqueeze(0), flat.unsqueeze(0)).item()
            # Clip negative similarities to 0 or epsilon to avoid subtracting?
            # Or use Softmax? Paper dependent. We use ReLU (clipped at 0)
            sims.append(max(0.0, cos_sim))

        total_sim = sum(sims)
        if total_sim < 1e-9:
            # Fallback to mean if all orthogonal/negative
            weights = [1.0 / len(updates)] * len(updates)
        else:
            weights = [s / total_sim for s in sims]

        # Aggregate
        # Reconstruct new global from weighted sum of flats
        new_flat = torch.zeros_like(global_vec)
        for w, flat in zip(weights, flat_updates):
            new_flat += w * flat

        # Unflatten
        new_state_dict = self._unflatten(new_flat, current_global_model.state_dict())
        current_global_model.load_state_dict(new_state_dict)

        return current_global_model, {"avg_similarity": float(sum(sims)/len(sims))}

    def _flatten(self, state_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        # Sort keys to ensure order
        keys = sorted(state_dict.keys())
        tensors = [state_dict[k].float().view(-1) for k in keys]
        return torch.cat(tensors)

    def _unflatten(self, flat: torch.Tensor, reference_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        keys = sorted(reference_dict.keys())
        output = {}
        offset = 0
        for k in keys:
            ref = reference_dict[k]
            numel = ref.numel()
            chunk = flat[offset : offset + numel]
            output[k] = chunk.view(ref.shape).to(ref.dtype)
            offset += numel
        return output
