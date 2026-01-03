"""
Standard dataset loaders.
"""

from typing import Tuple, Any
from unbitrium.datasets.registry import DatasetRegistry

# Placeholder for torch imports
try:
    import torch
    from torchvision import datasets, transforms
except ImportError:
    torch = None

def load_dataset(name: str, cache_dir: str = "./data") -> Tuple[Any, Any]:
    """
    Load a dataset by name.

    Returns
    -------
    Tuple[Any, Any]
        (train_dataset, test_dataset)
    """
    loader = DatasetRegistry.get(name)
    return loader(cache_dir)

@DatasetRegistry.register("cifar10")
def load_cifar10(cache_dir: str) -> Tuple[Any, Any]:
    if not torch:
        raise ImportError("PyTorch required for CIFAR-10")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_set = datasets.CIFAR10(root=cache_dir, train=True, download=True, transform=transform)
    test_set = datasets.CIFAR10(root=cache_dir, train=False, download=True, transform=transform)
    return train_set, test_set

@DatasetRegistry.register("mnist")
def load_mnist(cache_dir: str) -> Tuple[Any, Any]:
    if not torch:
        raise ImportError("PyTorch required for MNIST")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_set = datasets.MNIST(root=cache_dir, train=True, download=True, transform=transform)
    test_set = datasets.MNIST(root=cache_dir, train=False, download=True, transform=transform)
    return train_set, test_set
