"""Dataset registry for Unbitrium.

Provides a centralized registry for dataset loading and configuration.

Author: Olaf Yunus Laitinen Imanov <oyli@dtu.dk>
License: EUPL-1.2
"""

from __future__ import annotations

from typing import Any, Callable


class DatasetRegistry:
    """Registry for federated learning datasets.

    Provides a centralized mechanism for registering and retrieving
    dataset loaders with standardized interfaces.

    Example:
        >>> @DatasetRegistry.register("my_dataset")
        ... def load_my_dataset(root: str, **kwargs):
        ...     return MyDataset(root)
        >>> dataset = DatasetRegistry.get("my_dataset", root="./data")
    """

    _registry: dict[str, Callable[..., Any]] = {}

    @classmethod
    def register(cls, name: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Decorator to register a dataset loader.

        Args:
            name: Dataset name.

        Returns:
            Decorator function.
        """

        def decorator(loader: Callable[..., Any]) -> Callable[..., Any]:
            cls._registry[name.lower()] = loader
            return loader

        return decorator

    @classmethod
    def get(cls, name: str, **kwargs: Any) -> Any:
        """Get dataset by name.

        Args:
            name: Dataset name.
            **kwargs: Arguments passed to dataset loader.

        Returns:
            Dataset instance.

        Raises:
            KeyError: If dataset is not registered.
        """
        if name.lower() not in cls._registry:
            available = ", ".join(cls._registry.keys())
            raise KeyError(f"Dataset '{name}' not found. Available: {available}")
        return cls._registry[name.lower()](**kwargs)

    @classmethod
    def list_datasets(cls) -> list[str]:
        """List registered dataset names.

        Returns:
            List of dataset names.
        """
        return list(cls._registry.keys())


# Convenience functions
def register_dataset(name: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Register a dataset loader.

    Args:
        name: Dataset name.

    Returns:
        Decorator function.
    """
    return DatasetRegistry.register(name)


def get_dataset(name: str, **kwargs: Any) -> Any:
    """Get dataset by name.

    Args:
        name: Dataset name.
        **kwargs: Arguments passed to loader.

    Returns:
        Dataset instance.
    """
    return DatasetRegistry.get(name, **kwargs)


# Register built-in datasets
@DatasetRegistry.register("synthetic")
def _load_synthetic(
    num_samples: int = 1000,
    num_features: int = 20,
    num_classes: int = 10,
    seed: int = 42,
) -> tuple[Any, Any]:
    """Load synthetic classification dataset."""
    import torch

    torch.manual_seed(seed)
    X = torch.randn(num_samples, num_features)
    y = torch.randint(0, num_classes, (num_samples,))
    return X, y
