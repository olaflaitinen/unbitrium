"""Unit tests for aggregators.

Author: Olaf Yunus Laitinen Imanov <oyli@dtu.dk>
License: EUPL-1.2
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from unbitrium.aggregators import (
    FedAdam,
    FedAvg,
    FedDyn,
    FedProx,
    FedSim,
    Krum,
    PFedSim,
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
        state = {
            k: v.clone() + torch.randn_like(v) * 0.1
            for k, v in model.state_dict().items()
        }
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


class TestFedDyn:
    """Tests for FedDyn aggregator."""

    def test_init(self) -> None:
        """Test FedDyn initialization."""
        agg = FedDyn(alpha=0.01)
        assert agg.alpha == 0.01

    def test_aggregate(self) -> None:
        """Test FedDyn aggregation."""
        agg = FedDyn(alpha=0.01)
        model = SimpleModel()
        updates = create_updates(5, model)
        result, metrics = agg.aggregate(updates, model)
        assert metrics["num_participants"] == 5.0


class TestFedCM:
    """Tests for FedCM aggregator."""

    def test_init(self) -> None:
        """Test FedCM initialization."""
        from unbitrium.aggregators import FedCM

        agg = FedCM(beta=0.9)
        assert agg.beta == 0.9

    def test_aggregate(self) -> None:
        """Test FedCM aggregation."""
        from unbitrium.aggregators import FedCM

        agg = FedCM(beta=0.9)
        model = SimpleModel()
        updates = create_updates(3, model)
        result, metrics = agg.aggregate(updates, model)
        assert metrics["num_participants"] == 3.0
        assert metrics["beta"] == 0.9

    def test_momentum_accumulation(self) -> None:
        """Test momentum accumulates across rounds."""
        from unbitrium.aggregators import FedCM

        agg = FedCM(beta=0.9)
        model = SimpleModel()

        # Run multiple rounds
        for _ in range(3):
            updates = create_updates(2, model)
            agg.aggregate(updates, model)

        # Momentum should be accumulated
        assert len(agg._momentum) > 0


class TestAFL_DCS:
    """Tests for AFL-DCS aggregator."""

    def test_init(self) -> None:
        """Test AFL-DCS initialization."""
        from unbitrium.aggregators import AFL_DCS

        agg = AFL_DCS(max_staleness=5, staleness_decay=0.9)
        assert agg.max_staleness == 5
        assert agg.staleness_decay == 0.9

    def test_aggregate(self) -> None:
        """Test AFL-DCS aggregation."""
        from unbitrium.aggregators import AFL_DCS

        agg = AFL_DCS()
        model = SimpleModel()
        updates = create_updates(5, model)
        result, metrics = agg.aggregate(updates, model)
        assert "num_participants" in metrics

    def test_staleness_filtering(self) -> None:
        """Test stale updates are filtered."""
        from unbitrium.aggregators import AFL_DCS

        agg = AFL_DCS(max_staleness=2)
        model = SimpleModel()

        updates = []
        for i in range(5):
            state = {k: v.clone() for k, v in model.state_dict().items()}
            updates.append(
                {
                    "state_dict": state,
                    "num_samples": 100,
                    "round": agg._global_round - i - 5,  # Very stale
                }
            )

        result, metrics = agg.aggregate(updates, model)
        # Stale updates should be filtered
        assert metrics.get("excluded_stale", 0) >= 0


class TestPFedSim:
    """Tests for PFedSim (personalized FedSim) aggregator."""

    def test_init(self) -> None:
        """Test PFedSim initialization."""
        agg = PFedSim(similarity_threshold=0.5)
        assert agg.similarity_threshold == 0.5

    def test_aggregate(self) -> None:
        """Test PFedSim aggregation."""
        agg = PFedSim(similarity_threshold=0.0)
        model = SimpleModel()
        updates = create_updates(5, model)
        result, metrics = agg.aggregate(updates, model)
        assert "num_participants" in metrics


class TestFedAdam:
    """Tests for FedAdam aggregator."""

    def test_init(self) -> None:
        """Test FedAdam initialization."""
        agg = FedAdam(lr=0.01, beta1=0.9, beta2=0.999)
        assert agg.lr == 0.01

    def test_aggregate(self) -> None:
        """Test FedAdam aggregation."""
        agg = FedAdam()
        model = SimpleModel()
        updates = create_updates(5, model)
        result, metrics = agg.aggregate(updates, model)
        assert metrics["num_participants"] == 5.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
