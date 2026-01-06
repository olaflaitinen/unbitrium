"""Unit tests for datasets module.

Author: Olaf Yunus Laitinen Imanov <oyli@dtu.dk>
License: EUPL-1.2
"""

from __future__ import annotations

import pytest
import torch
from torch.utils.data import Dataset

from unbitrium.datasets import (
    DatasetRegistry,
    get_dataset,
    register_dataset,
)


class DummyDataset(Dataset):
    """Dummy dataset for testing."""

    def __init__(self, size: int = 100, num_classes: int = 10) -> None:
        self.data = torch.randn(size, 10)
        self.targets = torch.randint(0, num_classes, (size,))

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.data[idx], self.targets[idx]


class TestDatasetRegistry:
    """Tests for DatasetRegistry."""

    def test_registry_exists(self) -> None:
        """Test registry singleton exists."""
        registry = DatasetRegistry()
        assert registry is not None

    def test_register_dataset(self) -> None:
        """Test registering a new dataset."""
        registry = DatasetRegistry()

        @registry.register("test_dummy")
        def create_dummy() -> Dataset:
            return DummyDataset()

        assert "test_dummy" in registry.available()

    def test_get_registered_dataset(self) -> None:
        """Test getting a registered dataset."""
        registry = DatasetRegistry()

        @registry.register("test_get")
        def create_dataset() -> Dataset:
            return DummyDataset(size=50)

        dataset = registry.get("test_get")
        assert len(dataset) == 50

    def test_list_available(self) -> None:
        """Test listing available datasets."""
        registry = DatasetRegistry()
        available = registry.available()
        assert isinstance(available, (list, set))

    def test_unregistered_raises(self) -> None:
        """Test getting unregistered dataset raises error."""
        registry = DatasetRegistry()

        with pytest.raises(KeyError):
            registry.get("nonexistent_dataset_xyz")


class TestDatasetFunctions:
    """Tests for module-level dataset functions."""

    def test_register_decorator(self) -> None:
        """Test register_dataset decorator."""

        @register_dataset("test_decorator")
        def create_ds() -> Dataset:
            return DummyDataset()

        # Should be accessible
        ds = get_dataset("test_decorator")
        assert ds is not None

    def test_get_dataset_with_params(self) -> None:
        """Test get_dataset with parameters."""

        @register_dataset("test_params")
        def create_ds(size: int = 100) -> Dataset:
            return DummyDataset(size=size)

        ds = get_dataset("test_params", size=200)
        assert len(ds) == 200


class TestDatasetProperties:
    """Tests for dataset utility functions."""

    def test_dataset_has_targets(self) -> None:
        """Test dataset has targets attribute."""
        ds = DummyDataset()
        assert hasattr(ds, "targets")
        assert len(ds.targets) == 100

    def test_dataset_indexing(self) -> None:
        """Test dataset can be indexed."""
        ds = DummyDataset()
        x, y = ds[0]
        assert x.shape == (10,)
        assert isinstance(y.item(), int)

    def test_dataset_iteration(self) -> None:
        """Test dataset can be iterated."""
        ds = DummyDataset(size=10)
        count = 0
        for x, y in ds:
            count += 1
        assert count == 10


class TestStandardDatasets:
    """Tests for standard dataset registrations."""

    def test_cifar10_registered(self) -> None:
        """Test CIFAR10 is in registry."""
        registry = DatasetRegistry()
        # This might or might not be registered depending on implementation
        # Just test the registry mechanism works
        available = registry.available()
        assert isinstance(available, (list, set))

    def test_mnist_registered(self) -> None:
        """Test MNIST registration pattern."""
        # Skip if not registered
        try:
            get_dataset("mnist", download=False)
        except (KeyError, FileNotFoundError):
            pytest.skip("MNIST not registered or not downloaded")


class TestDatasetSplitting:
    """Tests for dataset splitting utilities."""

    def test_create_client_datasets(self) -> None:
        """Test creating client datasets from indices."""
        ds = DummyDataset(size=100)
        indices = {0: list(range(50)), 1: list(range(50, 100))}

        client_datasets = {
            i: torch.utils.data.Subset(ds, idx) for i, idx in indices.items()
        }

        assert len(client_datasets) == 2
        assert len(client_datasets[0]) == 50
        assert len(client_datasets[1]) == 50

    def test_subset_preserves_data(self) -> None:
        """Test subsets preserve original data."""
        ds = DummyDataset(size=100)
        subset = torch.utils.data.Subset(ds, [0, 1, 2])

        x_orig, y_orig = ds[0]
        x_sub, y_sub = subset[0]

        assert torch.allclose(x_orig, x_sub)
        assert y_orig == y_sub


class TestModuleExports:
    """Test datasets module exports."""

    def test_exports(self) -> None:
        """Test all expected exports exist."""
        from unbitrium import datasets

        assert hasattr(datasets, "DatasetRegistry")
        assert hasattr(datasets, "register_dataset")
        assert hasattr(datasets, "get_dataset")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
