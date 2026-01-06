# Getting Started

This guide will help you install Unbitrium and run your first federated learning experiment.

---

## Table of Contents

1. [Requirements](#requirements)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [Next Steps](#next-steps)

---

## Requirements

### System Requirements

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| Python | >= 3.10 | >= 3.12 |
| RAM | 8 GB | 16 GB |
| Storage | 1 GB | 5 GB |
| GPU | Optional | NVIDIA CUDA |

### Dependencies

Unbitrium requires the following core dependencies:

| Package | Version | Purpose |
|---------|---------|---------|
| torch | >= 2.0 | Deep learning framework |
| numpy | >= 2.0 | Numerical computing |
| scipy | >= 1.12 | Scientific computing |
| pyyaml | >= 6.0 | Configuration files |

---

## Installation

### From PyPI (Recommended)

```bash
pip install unbitrium
```

### From Source

```bash
git clone https://github.com/olaflaitinen/unbitrium.git
cd unbitrium
pip install -e .
```

### Development Installation

```bash
git clone https://github.com/olaflaitinen/unbitrium.git
cd unbitrium
pip install -e ".[dev,docs]"
pre-commit install
```

### Docker

```bash
docker pull olaflaitinen/unbitrium:latest
docker run -it unbitrium python -c "import unbitrium; print(unbitrium.__version__)"
```

---

## Quick Start

### Basic Example

```python
"""Minimal federated learning example with Unbitrium."""

import torch
import torch.nn as nn
import numpy as np

from unbitrium.aggregators import FedAvg
from unbitrium.partitioning import DirichletPartitioner
from unbitrium.metrics import compute_emd, compute_label_entropy
from unbitrium.simulation import Client

# Configuration
NUM_CLIENTS = 10
ALPHA = 0.5  # Dirichlet concentration (lower = more heterogeneous)
NUM_ROUNDS = 5
LOCAL_EPOCHS = 1

# Generate synthetic data
np.random.seed(42)
torch.manual_seed(42)

X = torch.randn(1000, 10)
y = torch.randint(0, 3, (1000,))

# Create partitioner
partitioner = DirichletPartitioner(num_clients=NUM_CLIENTS, alpha=ALPHA)

# Partition data
client_indices = partitioner.partition(y.numpy())

# Measure heterogeneity
emd = compute_emd(y.numpy(), client_indices)
entropy = compute_label_entropy(y.numpy(), client_indices)
print(f"EMD: {emd:.4f}, Entropy: {entropy:.4f}")

# Define model
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 3)

    def forward(self, x):
        return self.fc(x)

# Create global model
global_model = SimpleModel()

# Create clients
clients = []
for i in range(NUM_CLIENTS):
    indices = client_indices[i]
    client = Client(
        client_id=i,
        local_data=(X[indices], y[indices]),
        model_fn=SimpleModel,
        local_epochs=LOCAL_EPOCHS,
    )
    clients.append(client)

# Create aggregator
aggregator = FedAvg()

# Training loop
for round_num in range(NUM_ROUNDS):
    # Collect updates
    updates = []
    for client in clients:
        update = client.train(global_model.state_dict())
        updates.append(update)

    # Aggregate
    global_model, metrics = aggregator.aggregate(updates, global_model)
    print(f"Round {round_num + 1}: {metrics}")

print("Training complete!")
```

### Verify Installation

```python
import unbitrium

print(f"Version: {unbitrium.__version__}")
print(f"Author: {unbitrium.__author__}")

# List available aggregators
from unbitrium.aggregators import __all__ as aggregators
print(f"Aggregators: {aggregators}")

# List available partitioners
from unbitrium.partitioning import __all__ as partitioners
print(f"Partitioners: {partitioners}")
```

---

## Next Steps

| Topic | Description |
|-------|-------------|
| [Tutorials](../tutorials/index.md) | 200+ comprehensive tutorials |
| [API Reference](../api/core.md) | Complete API documentation |
| [Examples](https://github.com/olaflaitinen/unbitrium/tree/main/examples) | Example scripts |
| [Contributing](../../CONTRIBUTING.md) | How to contribute |

---

## Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| ImportError | Ensure package is installed: `pip install unbitrium` |
| CUDA not found | Install PyTorch with CUDA: `pip install torch --index-url https://download.pytorch.org/whl/cu118` |
| Type errors | Run mypy: `mypy src/` |

### Getting Help

- [Documentation](https://olaflaitinen.github.io/unbitrium/)
- [GitHub Issues](https://github.com/olaflaitinen/unbitrium/issues)
- [Discussions](https://github.com/olaflaitinen/unbitrium/discussions)

---

*Last updated: January 2026*
