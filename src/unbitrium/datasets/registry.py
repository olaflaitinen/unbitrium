
from typing import Dict, Any, Tuple, Optional
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision

class DatasetRegistry:
    _registry = {}

    @classmethod
    def register(cls, name: str):
        def decorator(dataset_cls):
            cls._registry[name] = dataset_cls
            return dataset_cls
        return decorator

    @classmethod
    def get(cls, name: str, **kwargs) -> Dataset:
        if name not in cls._registry:
            raise ValueError(f"Dataset {name} not found in registry")
        return cls._registry[name](**kwargs)

@DatasetRegistry.register("cifar10")
class CIFAR10Dataset(Dataset):
    def __init__(self, root: str = "./data", train: bool = True, download: bool = True):
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.data = torchvision.datasets.CIFAR10(
            root=root, train=train, download=download, transform=transform
        )

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

@DatasetRegistry.register("mnist")
class MNISTDataset(Dataset):
    def __init__(self, root: str = "./data", train: bool = True, download: bool = True):
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.data = torchvision.datasets.MNIST(
            root=root, train=train, download=download, transform=transform
        )

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)
