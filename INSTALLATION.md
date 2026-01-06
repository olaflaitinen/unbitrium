# Installation Guide

This document provides comprehensive installation instructions for Unbitrium.

---

## Table of Contents

1. [Requirements](#requirements)
2. [Quick Installation](#quick-installation)
3. [Development Installation](#development-installation)
4. [GPU Support](#gpu-support)
5. [Optional Dependencies](#optional-dependencies)
6. [Docker Installation](#docker-installation)
7. [Verification](#verification)
8. [Troubleshooting](#troubleshooting)

---

## Requirements

### System Requirements

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| Python | 3.10 | 3.12+ |
| RAM | 8 GB | 16 GB |
| Storage | 1 GB | 5 GB |
| GPU | Optional | NVIDIA CUDA |

### Operating Systems

| OS | Supported |
|----|-----------|
| Ubuntu 20.04+ | Yes |
| Debian 11+ | Yes |
| macOS 12+ | Yes |
| Windows 10/11 | Yes |

---

## Quick Installation

### From PyPI

```bash
pip install unbitrium
```

### From Source

```bash
git clone https://github.com/olaflaitinen/unbitrium.git
cd unbitrium
pip install .
```

---

## Development Installation

### Prerequisites

```bash
# Ensure Python 3.10+
python --version

# Ensure pip is up to date
pip install --upgrade pip
```

### Clone and Install

```bash
# Clone repository
git clone https://github.com/olaflaitinen/unbitrium.git
cd unbitrium

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On Windows:
.venv\Scripts\activate
# On Unix/macOS:
source .venv/bin/activate

# Install in development mode with all extras
pip install -e ".[dev,docs]"
```

### Pre-commit Hooks

```bash
# Install pre-commit hooks
pre-commit install

# Verify installation
pre-commit run --all-files
```

---

## GPU Support

### NVIDIA CUDA (Recommended)

1. Install NVIDIA drivers for your GPU
2. Install CUDA Toolkit (11.8+)
3. Install cuDNN

```bash
# Install PyTorch with CUDA support
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### Verification

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"GPU name: {torch.cuda.get_device_name(0)}")
```

### Apple Silicon (M1/M2/M3)

PyTorch supports MPS (Metal Performance Shaders) on Apple Silicon:

```python
import torch
print(f"MPS available: {torch.backends.mps.is_available()}")
```

---

## Optional Dependencies

### Core Extras

```bash
# All development dependencies
pip install -e ".[dev]"

# Documentation dependencies
pip install -e ".[docs]"

# Visualization dependencies
pip install -e ".[viz]"
```

### Available Extras

| Extra | Dependencies |
|-------|--------------|
| `dev` | pytest, black, isort, mypy, ruff, pre-commit |
| `docs` | mkdocs, mkdocs-material, mkdocstrings |
| `viz` | matplotlib, seaborn, plotly |
| `all` | All of the above |

### Installation Examples

```bash
# Developer installation
pip install -e ".[dev]"

# Full installation
pip install -e ".[all]"

# Production with GPU
pip install unbitrium torch --index-url https://download.pytorch.org/whl/cu118
```

---

## Docker Installation

### Pull Image

```bash
docker pull olaflaitinen/unbitrium:latest
```

### Build from Source

```bash
# Build image
docker build -t unbitrium:latest .

# Run container
docker run -it --gpus all unbitrium:latest python -c "import unbitrium; print(unbitrium.__version__)"
```

### Docker Compose

```yaml
version: '3.8'
services:
  unbitrium:
    image: olaflaitinen/unbitrium:latest
    volumes:
      - ./data:/app/data
      - ./results:/app/results
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

---

## Verification

### Basic Check

```python
import unbitrium

print(f"Unbitrium version: {unbitrium.__version__}")
```

### Component Check

```python
from unbitrium.aggregators import FedAvg
from unbitrium.partitioning import DirichletPartitioner
from unbitrium.metrics import compute_emd

print("All components imported successfully!")
```

### Run Tests

```bash
pytest tests/test_smoke.py -v
```

### Full Verification

```bash
# Run all tests
pytest

# Type checking
mypy src/

# Linting
ruff check src/
```

---

## Troubleshooting

### Common Issues

#### ImportError: No module named 'unbitrium'

Ensure the package is installed:

```bash
pip install unbitrium
# Or for development:
pip install -e .
```

#### Torch not found

Install PyTorch separately:

```bash
pip install torch
```

#### CUDA not available

1. Verify NVIDIA driver installation
2. Check CUDA toolkit installation
3. Ensure PyTorch was installed with CUDA support

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

#### Permission denied

Use a virtual environment or `--user` flag:

```bash
pip install --user unbitrium
```

### Getting Help

1. Check [documentation](https://olaflaitinen.github.io/unbitrium/)
2. Search [issues](https://github.com/olaflaitinen/unbitrium/issues)
3. See [SUPPORT.md](SUPPORT.md)

---

*Last updated: January 2026*
