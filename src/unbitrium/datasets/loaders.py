"""Standard dataset loaders for Unbitrium.

Provides preregistered loaders for common federated learning benchmarks
including CIFAR-10, MNIST, and synthetic datasets.

Author: Olaf Yunus Laitinen Imanov <oyli@dtu.dk>
License: EUPL-1.2
"""

from __future__ import annotations

from typing import Any, Tuple

from unbitrium.datasets.registry import DatasetRegistry

try:
    import torch
    from torchvision import datasets, transforms
    _TORCH_AVAILABLE = True
except ImportError:
    torch = None  # type: ignore
    _TORCH_AVAILABLE = False


def load_dataset(name: str, cache_dir: str = "./data") -> Tuple[Any, Any]:
    """Load a dataset by name from the registry.

    Args:
        name: Registered dataset name (e.g., 'cifar10', 'mnist').
        cache_dir: Directory for caching downloaded data.

    Returns:
        Tuple of (train_dataset, test_dataset).

    Raises:
        KeyError: If dataset name is not registered.

    Example:
        >>> train, test = load_dataset("cifar10", cache_dir="./data")
    """
    loader = DatasetRegistry.get(name)
    return loader(cache_dir)


@DatasetRegistry.register("cifar10")
def load_cifar10(cache_dir: str = "./data") -> Tuple[Any, Any]:
    """Load CIFAR-10 dataset with standard normalization.

    Args:
        cache_dir: Directory for caching downloaded data.

    Returns:
        Tuple of (train_dataset, test_dataset).

    Raises:
        ImportError: If PyTorch/torchvision is not available.
    """
    if not _TORCH_AVAILABLE:
        raise ImportError("PyTorch and torchvision required for CIFAR-10")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])

    train_set = datasets.CIFAR10(
        root=cache_dir, train=True, download=True, transform=transform
    )
    test_set = datasets.CIFAR10(
        root=cache_dir, train=False, download=True, transform=transform
    )
    return train_set, test_set


@DatasetRegistry.register("cifar100")
def load_cifar100(cache_dir: str = "./data") -> Tuple[Any, Any]:
    """Load CIFAR-100 dataset with standard normalization.

    Args:
        cache_dir: Directory for caching downloaded data.

    Returns:
        Tuple of (train_dataset, test_dataset).

    Raises:
        ImportError: If PyTorch/torchvision is not available.
    """
    if not _TORCH_AVAILABLE:
        raise ImportError("PyTorch and torchvision required for CIFAR-100")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

    train_set = datasets.CIFAR100(
        root=cache_dir, train=True, download=True, transform=transform
    )
    test_set = datasets.CIFAR100(
        root=cache_dir, train=False, download=True, transform=transform
    )
    return train_set, test_set


@DatasetRegistry.register("mnist")
def load_mnist(cache_dir: str = "./data") -> Tuple[Any, Any]:
    """Load MNIST dataset with standard normalization.

    Args:
        cache_dir: Directory for caching downloaded data.

    Returns:
        Tuple of (train_dataset, test_dataset).

    Raises:
        ImportError: If PyTorch/torchvision is not available.
    """
    if not _TORCH_AVAILABLE:
        raise ImportError("PyTorch and torchvision required for MNIST")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    train_set = datasets.MNIST(
        root=cache_dir, train=True, download=True, transform=transform
    )
    test_set = datasets.MNIST(
        root=cache_dir, train=False, download=True, transform=transform
    )
    return train_set, test_set


@DatasetRegistry.register("femnist")
def load_femnist(cache_dir: str = "./data") -> Tuple[Any, Any]:
    """Load Federated EMNIST dataset.

    FEMNIST is a federated version of EMNIST commonly used
    for federated learning benchmarks.

    Args:
        cache_dir: Directory for caching downloaded data.

    Returns:
        Tuple of (train_dataset, test_dataset).

    Raises:
        ImportError: If PyTorch/torchvision is not available.
    """
    if not _TORCH_AVAILABLE:
        raise ImportError("PyTorch and torchvision required for FEMNIST")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    train_set = datasets.EMNIST(
        root=cache_dir, split="byclass", train=True, download=True, transform=transform
    )
    test_set = datasets.EMNIST(
        root=cache_dir, split="byclass", train=False, download=True, transform=transform
    )
    return train_set, test_set
