"""Unit tests for metrics.

Author: Olaf Yunus Laitinen Imanov <oyli@dtu.dk>
License: EUPL-1.2
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from unbitrium.metrics import (
    compute_distribution_metrics,
    compute_drift_norm,
    compute_emd,
    compute_fairness_metrics,
    compute_gradient_variance,
    compute_imbalance_ratio,
    compute_js_divergence,
    compute_label_entropy,
)


def create_partition(
    num_samples: int = 1000,
    num_clients: int = 10,
    num_classes: int = 10,
) -> tuple[np.ndarray, dict[int, list[int]]]:
    """Create mock partition."""
    labels = np.random.randint(0, num_classes, num_samples)
    indices = {i: list(range(i * 100, (i + 1) * 100)) for i in range(num_clients)}
    return labels, indices


class TestHeterogeneityMetrics:
    """Tests for heterogeneity metrics."""

    def test_label_entropy(self) -> None:
        """Test label entropy computation."""
        labels, indices = create_partition()
        entropy = compute_label_entropy(labels, indices)
        assert entropy >= 0
        assert entropy <= np.log(10)  # Max entropy for 10 classes

    def test_emd(self) -> None:
        """Test EMD computation."""
        labels, indices = create_partition()
        emd = compute_emd(labels, indices)
        assert emd >= 0
        assert emd <= 2  # L1 distance bounded by 2

    def test_js_divergence(self) -> None:
        """Test JS divergence."""
        labels, indices = create_partition()
        js = compute_js_divergence(labels, indices)
        assert js >= 0
        assert js <= np.log(2)  # Max JS divergence

    def test_imbalance_ratio(self) -> None:
        """Test imbalance ratio."""
        counts = [100, 200, 50, 150]
        ratio = compute_imbalance_ratio(counts)
        assert ratio == 4.0  # 200 / 50


class TestGradientMetrics:
    """Tests for gradient-related metrics."""

    def test_gradient_variance(self) -> None:
        """Test gradient variance computation."""
        global_model = {"fc.weight": torch.randn(10, 5)}
        local_models = [
            {"fc.weight": global_model["fc.weight"] + torch.randn(10, 5) * 0.1}
            for _ in range(5)
        ]
        variance = compute_gradient_variance(local_models, global_model)
        assert variance > 0

    def test_drift_norm(self) -> None:
        """Test drift norm computation."""
        initial = {"fc.weight": torch.zeros(10, 5)}
        final = {"fc.weight": torch.ones(10, 5)}
        norm = compute_drift_norm(initial, final)
        assert norm == pytest.approx(np.sqrt(50), rel=0.01)


class TestDistributionMetrics:
    """Tests for distribution metrics."""

    def test_compute_distribution_metrics(self) -> None:
        """Test comprehensive distribution metrics."""
        labels, indices = create_partition()
        metrics = compute_distribution_metrics(labels, indices)

        assert "num_clients" in metrics
        assert "num_classes" in metrics
        assert "total_samples" in metrics
        assert "avg_l1_distance" in metrics


class TestFairnessMetrics:
    """Tests for fairness metrics."""

    def test_compute_fairness_metrics(self) -> None:
        """Test fairness metrics computation."""
        accuracies = [0.8, 0.85, 0.75, 0.9, 0.7]
        metrics = compute_fairness_metrics(accuracies)

        assert metrics["min_accuracy"] == 0.7
        assert metrics["max_accuracy"] == 0.9
        assert "jains_fairness_index" in metrics


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
