import torch
from typing import List, Optional

class GaussianMechanism:
    """
    Implements the Gaussian Mechanism for Differential Privacy.
    """
    def __init__(self, noise_multiplier: float, max_grad_norm: float):
        self.noise_multiplier = noise_multiplier
        self.max_grad_norm = max_grad_norm

    def add_noise(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Adds Gaussian noise to the tensor after clipping.
        """
        # Clip
        norm = tensor.norm(2)
        clip_coef = self.max_grad_norm / (norm + 1e-6)
        clip_coef = torch.min(clip_coef, torch.ones_like(clip_coef))
        clipped_tensor = tensor * clip_coef

        # Add noise
        noise_std = self.noise_multiplier * self.max_grad_norm
        noise = torch.randn_like(tensor) * noise_std

        return clipped_tensor + noise

    def get_privacy_spent(self, delta: float) -> float:
        """
        Placeholder for privacy accounting based on steps/rounds.
        Full RDP implementation would go here.
        """
        # simplified
        return 0.0
