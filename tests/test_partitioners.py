"""Unit tests for partitioners.

Author: Olaf Yunus Laitinen Imanov <oyli@dtu.dk>
License: EUPL-1.2
"""

from __future__ import annotations

import numpy as np
import pytest

from unbitrium.partitioning import (
    DirichletPartitioner,
    EntropyControlledPartitioner,
    FeatureShiftPartitioner,
    MoDMPartitioner,
    QuantitySkewPartitioner,
)


def create_labels(num_samples: int = 1000, num_classes: int = 10) -> np.ndarray:
    """Create mock labels."""
    np.random.seed(42)
    return np.random.randint(0, num_classes, num_samples)


class TestDirichletPartitioner:
    """Tests for Dirichlet partitioner."""

    def test_partition_coverage(self) -> None:
        """Test all samples are assigned."""
        labels = create_labels()
        partitioner = DirichletPartitioner(num_clients=10, alpha=0.5)
        indices = partitioner.partition(labels)

        total = sum(len(idx) for idx in indices.values())
        assert total == len(labels)

    def test_partition_low_alpha(self) -> None:
        """Test low alpha creates more skew."""
        labels = create_labels()
        partitioner = DirichletPartitioner(num_clients=10, alpha=0.1)
        indices = partitioner.partition(labels)
        assert len(indices) == 10

    def test_partition_high_alpha(self) -> None:
        """Test high alpha creates more uniform."""
        labels = create_labels()
        partitioner = DirichletPartitioner(num_clients=10, alpha=10.0)
        indices = partitioner.partition(labels)
        sizes = [len(idx) for idx in indices.values()]
        # Higher alpha should have more balanced sizes
        assert np.std(sizes) < 100


class TestMoDMPartitioner:
    """Tests for MoDM partitioner."""

    def test_partition(self) -> None:
        """Test MoDM partitioning."""
        labels = create_labels()
        partitioner = MoDMPartitioner(num_clients=20, num_modes=3)
        indices = partitioner.partition(labels)

        total = sum(len(idx) for idx in indices.values())
        assert total == len(labels)


class TestQuantitySkewPartitioner:
    """Tests for Quantity Skew partitioner."""

    def test_partition(self) -> None:
        """Test quantity skew."""
        labels = create_labels()
        partitioner = QuantitySkewPartitioner(num_clients=10, gamma=1.5)
        indices = partitioner.partition(labels)

        sizes = [len(idx) for idx in indices.values()]
        # First client should have more samples
        assert sizes[0] > sizes[-1]


class TestEntropyControlledPartitioner:
    """Tests for Entropy Controlled partitioner."""

    def test_partition(self) -> None:
        """Test entropy controlled partitioning."""
        labels = create_labels()
        partitioner = EntropyControlledPartitioner(num_clients=10, target_entropy=1.5)
        indices = partitioner.partition(labels)

        total = sum(len(idx) for idx in indices.values())
        assert total == len(labels)


class TestFeatureShiftPartitioner:
    """Tests for Feature Shift partitioner."""

    def test_partition(self) -> None:
        """Test feature shift partitioning."""
        features = np.random.randn(1000, 20)
        partitioner = FeatureShiftPartitioner(num_clients=10, num_clusters=5)
        indices = partitioner.partition(features)

        total = sum(len(idx) for idx in indices.values())
        assert total == len(features)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
