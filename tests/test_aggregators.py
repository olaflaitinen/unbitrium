"""Unit tests for aggregators.

Author: Olaf Yunus Laitinen Imanov <oyli@dtu.dk>
License: EUPL-1.2
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from unbitrium.aggregators import (
    FedAvg,
    FedProx,
    FedSim,
    PFedSim,
    FedDyn,
    FedAdam,
    Krum,
    TrimmedMean,
)


class SimpleModel(nn.Module):
    """Simple model for testing."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(10, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


def create_updates(num_clients: int, model: nn.Module) -> list[dict]:
    """Create mock client updates."""
    updates = []
    for i in range(num_clients):
        state = {k: v.clone() + torch.randn_like(v) * 0.1 for k, v in model.state_dict().items()}
        updates.append({"state_dict": state, "num_samples": 100 + i * 10})
    return updates


class TestFedAvg:
    """Tests for FedAvg aggregator."""

    def test_aggregate_empty(self) -> None:
        """Test with empty updates."""
        agg = FedAvg()
        model = SimpleModel()
        result, metrics = agg.aggregate([], model)
        assert metrics["aggregated_clients"] == 0.0

    def test_aggregate_single_client(self) -> None:
        """Test with single client."""
        agg = FedAvg()
        model = SimpleModel()
        updates = create_updates(1, model)
        result, metrics = agg.aggregate(updates, model)
        assert metrics["num_participants"] == 1.0

    def test_aggregate_multiple_clients(self) -> None:
        """Test with multiple clients."""
        agg = FedAvg()
        model = SimpleModel()
        updates = create_updates(5, model)
        result, metrics = agg.aggregate(updates, model)
        assert metrics["num_participants"] == 5.0
        assert metrics["total_samples"] > 0


class TestFedProx:
    """Tests for FedProx aggregator."""

    def test_init_mu(self) -> None:
        """Test mu parameter."""
        agg = FedProx(mu=0.1)
        assert agg.mu == 0.1

    def test_aggregate(self) -> None:
        """Test aggregation."""
        agg = FedProx(mu=0.01)
        model = SimpleModel()
        updates = create_updates(3, model)
        result, metrics = agg.aggregate(updates, model)
        assert metrics["mu"] == 0.01


class TestFedSim:
    """Tests for FedSim aggregator."""

    def test_similarity_threshold(self) -> None:
        """Test similarity threshold filtering."""
        agg = FedSim(similarity_threshold=0.99)  # Very high threshold
        model = SimpleModel()
        updates = create_updates(5, model)
        result, metrics = agg.aggregate(updates, model)
        # Some clients should be filtered

    def test_aggregate(self) -> None:
        """Test normal aggregation."""
        agg = FedSim(similarity_threshold=0.0)
        model = SimpleModel()
        updates = create_updates(3, model)
        result, metrics = agg.aggregate(updates, model)
        assert "avg_similarity" in metrics


class TestKrum:
    """Tests for Krum aggregator."""

    def test_aggregate(self) -> None:
        """Test Krum aggregation."""
        agg = Krum(num_byzantine=0)
        model = SimpleModel()
        updates = create_updates(5, model)
        result, metrics = agg.aggregate(updates, model)
        assert metrics["num_participants"] >= 1

    def test_multi_krum(self) -> None:
        """Test multi-Krum."""
        agg = Krum(num_byzantine=1, multi_krum=2)
        model = SimpleModel()
        updates = create_updates(10, model)
        result, metrics = agg.aggregate(updates, model)


class TestTrimmedMean:
    """Tests for TrimmedMean aggregator."""

    def test_aggregate(self) -> None:
        """Test trimmed mean aggregation."""
        agg = TrimmedMean(trim_ratio=0.1)
        model = SimpleModel()
        updates = create_updates(10, model)
        result, metrics = agg.aggregate(updates, model)
        assert metrics["effective_clients"] < 10

    def test_invalid_trim_ratio(self) -> None:
        """Test invalid trim ratio."""
        with pytest.raises(ValueError):
            TrimmedMean(trim_ratio=0.6)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
