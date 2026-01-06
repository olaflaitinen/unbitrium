"""Unit tests for privacy module.

Author: Olaf Yunus Laitinen Imanov <oyli@dtu.dk>
License: EUPL-1.2
"""

from __future__ import annotations

import pytest
import torch

from unbitrium.privacy import (
    GaussianMechanism,
    LaplaceMechanism,
    clip_gradients,
)


class TestGaussianMechanism:
    """Tests for Gaussian noise mechanism."""

    def test_init_valid_params(self) -> None:
        """Test initialization with valid parameters."""
        mech = GaussianMechanism(epsilon=1.0, delta=1e-5, sensitivity=1.0)
        assert mech.epsilon == 1.0
        assert mech.delta == 1e-5
        assert mech.sensitivity == 1.0

    def test_add_noise_shape_preserved(self) -> None:
        """Test noise addition preserves tensor shape."""
        mech = GaussianMechanism(epsilon=1.0, delta=1e-5, sensitivity=1.0)
        tensor = torch.randn(10, 5)
        noisy = mech.add_noise(tensor)
        assert noisy.shape == tensor.shape

    def test_add_noise_changes_values(self) -> None:
        """Test that noise is actually added."""
        mech = GaussianMechanism(epsilon=1.0, delta=1e-5, sensitivity=1.0)
        tensor = torch.ones(100)
        noisy = mech.add_noise(tensor)
        assert not torch.allclose(tensor, noisy)

    def test_noise_scale_increases_with_sensitivity(self) -> None:
        """Test more sensitivity means more noise."""
        mech_low = GaussianMechanism(epsilon=1.0, delta=1e-5, sensitivity=0.5)
        mech_high = GaussianMechanism(epsilon=1.0, delta=1e-5, sensitivity=2.0)

        tensor = torch.zeros(1000)
        noisy_low = mech_low.add_noise(tensor)
        noisy_high = mech_high.add_noise(tensor)

        # Higher sensitivity should have higher variance
        assert noisy_high.std() > noisy_low.std()

    def test_compute_noise_std(self) -> None:
        """Test noise standard deviation computation."""
        mech = GaussianMechanism(epsilon=1.0, delta=1e-5, sensitivity=1.0)
        sigma = mech.compute_noise_std()
        assert sigma > 0

    def test_privacy_budget_accounting(self) -> None:
        """Test privacy budget tracking."""
        mech = GaussianMechanism(epsilon=1.0, delta=1e-5, sensitivity=1.0)
        initial_spent = mech.get_spent_budget()
        mech.add_noise(torch.randn(10))
        after_spent = mech.get_spent_budget()
        assert after_spent >= initial_spent


class TestLaplaceMechanism:
    """Tests for Laplace noise mechanism."""

    def test_init(self) -> None:
        """Test initialization."""
        mech = LaplaceMechanism(epsilon=1.0, sensitivity=1.0)
        assert mech.epsilon == 1.0
        assert mech.sensitivity == 1.0

    def test_add_noise_shape(self) -> None:
        """Test noise preserves shape."""
        mech = LaplaceMechanism(epsilon=1.0, sensitivity=1.0)
        tensor = torch.randn(5, 10)
        noisy = mech.add_noise(tensor)
        assert noisy.shape == tensor.shape

    def test_noise_scale(self) -> None:
        """Test noise scale is sensitivity/epsilon."""
        mech = LaplaceMechanism(epsilon=2.0, sensitivity=4.0)
        assert mech.scale == 2.0  # 4.0 / 2.0

    def test_different_epsilon_different_noise(self) -> None:
        """Test different epsilon values produce different noise levels."""
        mech_tight = LaplaceMechanism(epsilon=0.1, sensitivity=1.0)
        mech_loose = LaplaceMechanism(epsilon=10.0, sensitivity=1.0)

        tensor = torch.zeros(10000)
        noisy_tight = mech_tight.add_noise(tensor)
        noisy_loose = mech_loose.add_noise(tensor)

        # Tighter privacy (smaller epsilon) means more noise
        assert noisy_tight.std() > noisy_loose.std()


class TestGradientClipping:
    """Tests for gradient clipping."""

    def test_clip_gradients_below_threshold(self) -> None:
        """Test gradients below threshold are unchanged."""
        grads = {"fc.weight": torch.ones(10, 5) * 0.1}
        clipped = clip_gradients(grads, max_norm=10.0)

        # Should be approximately the same
        assert torch.allclose(grads["fc.weight"], clipped["fc.weight"], atol=1e-5)

    def test_clip_gradients_above_threshold(self) -> None:
        """Test gradients above threshold are clipped."""
        grads = {"fc.weight": torch.ones(10, 5) * 10.0}
        clipped = clip_gradients(grads, max_norm=1.0)

        # Clipped norm should be <= max_norm
        total_norm = sum((v**2).sum() for v in clipped.values()) ** 0.5
        assert total_norm <= 1.0 + 1e-5

    def test_clip_gradients_preserves_direction(self) -> None:
        """Test clipping preserves gradient direction."""
        original = torch.tensor([3.0, 4.0])  # norm = 5
        grads = {"param": original}
        clipped = clip_gradients(grads, max_norm=1.0)

        # Direction should be preserved (normalized version)
        expected_direction = original / original.norm()
        actual_direction = clipped["param"] / clipped["param"].norm()
        assert torch.allclose(expected_direction, actual_direction, atol=1e-5)

    def test_clip_gradients_multiple_params(self) -> None:
        """Test clipping with multiple parameters."""
        grads = {
            "fc1.weight": torch.randn(10, 5) * 5,
            "fc1.bias": torch.randn(10) * 5,
            "fc2.weight": torch.randn(5, 2) * 5,
        }
        clipped = clip_gradients(grads, max_norm=1.0)

        assert len(clipped) == 3
        total_norm = sum((v**2).sum() for v in clipped.values()) ** 0.5
        assert total_norm <= 1.0 + 1e-5

    def test_clip_gradients_empty(self) -> None:
        """Test clipping empty gradient dict."""
        clipped = clip_gradients({}, max_norm=1.0)
        assert clipped == {}


class TestModuleExports:
    """Test privacy module exports."""

    def test_exports(self) -> None:
        """Test all expected exports exist."""
        from unbitrium import privacy

        assert hasattr(privacy, "GaussianMechanism")
        assert hasattr(privacy, "LaplaceMechanism")
        assert hasattr(privacy, "clip_gradients")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
