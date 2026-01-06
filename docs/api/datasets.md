# Datasets API Reference

This document provides the API reference for the `unbitrium.datasets` module.

---

## Table of Contents

1. [Loaders](#loaders)
2. [Registry](#registry)

---

## Loaders

### load_mnist

```python
from unbitrium.datasets import load_mnist

def load_mnist(
    root: str = "./data",
    train: bool = True,
    download: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load MNIST dataset.

    Args:
        root: Data directory.
        train: Load training set.
        download: Download if not present.

    Returns:
        Tuple of (images, labels).
    """
```

### load_cifar10

```python
from unbitrium.datasets import load_cifar10

def load_cifar10(
    root: str = "./data",
    train: bool = True,
    download: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load CIFAR-10 dataset."""
```

### load_cifar100

```python
from unbitrium.datasets import load_cifar100

def load_cifar100(
    root: str = "./data",
    train: bool = True,
    download: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load CIFAR-100 dataset."""
```

### load_femnist

```python
from unbitrium.datasets import load_femnist

def load_femnist(
    root: str = "./data",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load FEMNIST dataset from LEAF benchmark."""
```

---

## Registry

```python
from unbitrium.datasets import DatasetRegistry

class DatasetRegistry:
    """Registry for dataset loaders.

    Example:
        >>> registry = DatasetRegistry()
        >>> registry.register("custom", custom_loader)
        >>> data = registry.load("mnist")
    """
```

### Methods

| Method | Description |
|--------|-------------|
| `register(name, loader)` | Register custom loader |
| `load(name, **kwargs)` | Load dataset by name |
| `list()` | List available datasets |

---

*Last updated: January 2026*
