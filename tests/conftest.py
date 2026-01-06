"""Pytest configuration and shared fixtures.

Author: Olaf Yunus Laitinen Imanov <oyli@dtu.dk>
License: EUPL-1.2
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn


@pytest.fixture(autouse=True)
def set_seeds() -> None:
    """Set random seeds for reproducibility."""
    np.random.seed(42)
    torch.manual_seed(42)


@pytest.fixture
def simple_model() -> nn.Module:
    """Create a simple model for testing."""

    class SimpleModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc = nn.Linear(10, 2)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.fc(x)

    return SimpleModel()


@pytest.fixture
def sample_labels() -> np.ndarray:
    """Create sample labels for partitioning tests."""
    return np.random.randint(0, 10, 1000)


@pytest.fixture
def sample_features() -> np.ndarray:
    """Create sample features for testing."""
    return np.random.randn(1000, 20).astype(np.float32)


@pytest.fixture
def client_updates(simple_model: nn.Module) -> list[dict]:
    """Create mock client updates."""
    updates = []
    for i in range(5):
        state = {
            k: v.clone() + torch.randn_like(v) * 0.1
            for k, v in simple_model.state_dict().items()
        }
        updates.append({"state_dict": state, "num_samples": 100 + i * 10})
    return updates


@pytest.fixture
def global_model_state(simple_model: nn.Module) -> dict[str, torch.Tensor]:
    """Get global model state dictionary."""
    return {k: v.clone() for k, v in simple_model.state_dict().items()}


@pytest.fixture
def local_model_states(
    global_model_state: dict[str, torch.Tensor],
) -> list[dict[str, torch.Tensor]]:
    """Create variations of model states for gradient testing."""
    return [
        {
            k: v.clone() + torch.randn_like(v) * 0.1
            for k, v in global_model_state.items()
        }
        for _ in range(5)
    ]


@pytest.fixture
def sample_partition() -> tuple[np.ndarray, dict[int, list[int]]]:
    """Create a sample partition for testing."""
    labels = np.random.randint(0, 10, 1000)
    indices = {i: list(range(i * 100, (i + 1) * 100)) for i in range(10)}
    return labels, indices
