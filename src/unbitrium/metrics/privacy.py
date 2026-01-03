"""
Privacy Accounting.
"""

from typing import Optional
import math

class PrivacyAccountant:
    """
    Tracks privacy budget usage (epsilon, delta).
    Supports basic composition (Sequential).
    For advanced accounting (RDP, GDP), interaction with Opacus/TensorFlow Privacy is recommended.
    """

    def __init__(self, target_delta: float = 1e-5):
        self.target_delta = target_delta
        self.total_epsilon = 0.0
        # self.steps = []

    def step(self, noise_multiplier: float, sample_rate: float, steps: int = 1):
        """
        Updates budget for 'steps' iterations of mechanism with given parameters.
        Uses RDP-based accounting approximation or simple composition.

        Using Simple Moments Accountant approximation or Standard Composition (loose).
        For simplicity in this base class, we just log steps or use a placeholder calculation.
        """
        # Placeholder for simple composition:
        # eps = steps * sample_rate * ...
        # Real DP accounting is complex.
        # We will expose a method to 'estimate' current epsilon using Opacus logic if available.
        pass

    def get_epsilon(self) -> float:
        """
        Returns cuurent cumulative epsilon.
        """
        return self.total_epsilon

    def analyze_gaussian_mechanism(
        self,
        sample_rate: float,
        noise_multiplier: float,
        steps: int
    ) -> float:
        """
        Computes epsilon for Gaussian Mechanism using RDP conversion (standard modern approach).
        """
        # Very simplified RDP to DP conversion for standard Gaussian
        # Epsilon \approx q * sqrt(T * log(1/delta)) / sigma

        if noise_multiplier == 0:
            return float('inf')

        return sample_rate * math.sqrt(steps * math.log(1 / self.target_delta)) / noise_multiplier
