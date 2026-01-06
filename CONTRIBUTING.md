# Contributing to Unbitrium

Thank you for your interest in contributing to Unbitrium. This document provides comprehensive guidelines for contributing to the project, ensuring high-quality, maintainable code that advances federated learning research.

---

## Table of Contents

1. [Project Lead](#project-lead)
2. [Getting Started](#getting-started)
3. [Development Environment](#development-environment)
4. [Code Standards](#code-standards)
5. [Testing Guidelines](#testing-guidelines)
6. [Documentation Standards](#documentation-standards)
7. [Pull Request Process](#pull-request-process)
8. [Issue Guidelines](#issue-guidelines)
9. [Security Vulnerabilities](#security-vulnerabilities)
10. [Code of Conduct](#code-of-conduct)
11. [License](#license)

---

## Project Lead

Unbitrium is developed and maintained by **Olaf Yunus Laitinen Imanov** (<oyli@dtu.dk>), PhD Candidate at the Technical University of Denmark (DTU).

---

## Getting Started

### Prerequisites

| Requirement | Version | Purpose |
|-------------|---------|---------|
| Python | >= 3.10 | Runtime environment |
| Git | >= 2.30 | Version control |
| PyTorch | >= 2.0 | Deep learning backend |
| pip | >= 23.0 | Package management |

### Fork and Clone

1. **Fork the repository** on GitHub using the "Fork" button.

2. **Clone your fork** to your local machine:

   ```bash
   git clone https://github.com/YOUR-USERNAME/unbitrium.git
   cd unbitrium
   ```

3. **Add upstream remote** to keep your fork synchronized:

   ```bash
   git remote add upstream https://github.com/olaflaitinen/unbitrium.git
   ```

4. **Verify remotes**:

   ```bash
   git remote -v
   # origin    https://github.com/YOUR-USERNAME/unbitrium.git (fetch)
   # origin    https://github.com/YOUR-USERNAME/unbitrium.git (push)
   # upstream  https://github.com/olaflaitinen/unbitrium.git (fetch)
   # upstream  https://github.com/olaflaitinen/unbitrium.git (push)
   ```

---

## Development Environment

### Installation

Install the package in development mode with all optional dependencies:

```bash
# Core development installation
pip install -e ".[dev]"

# With documentation dependencies
pip install -e ".[dev,docs]"
```

### Pre-commit Hooks

We use pre-commit hooks to enforce code quality automatically:

```bash
# Install pre-commit hooks
pre-commit install

# Run all hooks manually
pre-commit run --all-files
```

### IDE Configuration

#### Visual Studio Code

Recommended extensions:
- Python (Microsoft)
- Pylance
- Black Formatter
- Ruff

Recommended `settings.json`:

```json
{
    "python.defaultInterpreterPath": ".venv/bin/python",
    "python.formatting.provider": "black",
    "editor.formatOnSave": true,
    "python.linting.enabled": true,
    "python.linting.mypyEnabled": true
}
```

#### PyCharm

Enable the following inspections:
- Type checking (strict mode)
- PEP 8 compliance
- Docstring validation

---

## Code Standards

### Python Style Guide

We adhere strictly to the [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html) with the following specifications:

| Aspect | Standard |
|--------|----------|
| Line length | 88 characters (Black default) |
| Indentation | 4 spaces |
| Quotes | Double quotes for strings |
| Imports | Grouped and sorted by isort |
| Docstrings | Google format |
| Type hints | Required for all functions |

### Formatting Tools

| Tool | Purpose | Configuration |
|------|---------|---------------|
| Black | Code formatting | `pyproject.toml` |
| isort | Import sorting | `pyproject.toml` |
| Ruff | Linting | `pyproject.toml` |
| mypy | Type checking | `pyproject.toml` |

Run formatting before committing:

```bash
# Format code
black src/ tests/

# Sort imports
isort src/ tests/

# Lint
ruff check src/ tests/

# Type check
mypy src/
```

### Module Structure

Each Python module must contain:

```python
"""Module description.

Extended description if needed.

Author: Olaf Yunus Laitinen Imanov <oyli@dtu.dk>
License: EUPL-1.2
"""

from __future__ import annotations

# Standard library imports
import os
import sys

# Third-party imports
import numpy as np
import torch

# Local imports
from unbitrium.core import utils
```

### Function Documentation

All public functions must have complete docstrings:

```python
def compute_gradient_variance(
    local_models: list[dict[str, torch.Tensor]],
    global_model: dict[str, torch.Tensor],
) -> float:
    """Compute variance of local model weights relative to global model.

    Mathematical formulation:

    $$
    \sigma^2 = \frac{1}{K} \sum_{k=1}^K \| w_k - w_g \|^2
    $$

    Args:
        local_models: List of client model state dictionaries.
        global_model: Global model state dictionary.

    Returns:
        Average squared L2 distance from global model.

    Raises:
        ValueError: If local_models is empty.

    Example:
        >>> variance = compute_gradient_variance(local_models, global_model)
        >>> print(f"Gradient variance: {variance:.4f}")
    """
```

---

## Testing Guidelines

### Test Structure

Tests are organized in the `tests/` directory mirroring the source structure:

```
tests/
    test_aggregators.py
    test_partitioners.py
    test_metrics.py
    test_integration.py
    conftest.py
```

### Writing Tests

Use pytest with the following conventions:

```python
"""Unit tests for aggregators.

Author: Olaf Yunus Laitinen Imanov <oyli@dtu.dk>
License: EUPL-1.2
"""

import pytest
import torch

from unbitrium.aggregators import FedAvg


class TestFedAvg:
    """Tests for FedAvg aggregator."""

    def test_aggregate_empty_updates(self) -> None:
        """Test aggregation with no client updates."""
        aggregator = FedAvg()
        model = SimpleModel()
        result, metrics = aggregator.aggregate([], model)
        assert metrics["aggregated_clients"] == 0.0

    @pytest.mark.parametrize("num_clients", [1, 5, 10, 100])
    def test_aggregate_various_client_counts(self, num_clients: int) -> None:
        """Test aggregation with varying client counts."""
        # Test implementation
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src/unbitrium --cov-report=html

# Run specific test file
pytest tests/test_aggregators.py

# Run with verbose output
pytest -v

# Run only fast tests (exclude slow markers)
pytest -m "not slow"
```

### Coverage Requirements

| Component | Minimum Coverage |
|-----------|------------------|
| Core modules | 80% |
| Aggregators | 90% |
| Metrics | 85% |
| Integration | 70% |

---

## Documentation Standards

### Docstring Format

Use Google-style docstrings consistently:

```python
def function(arg1: str, arg2: int = 10) -> bool:
    """Short description.

    Extended description if needed.

    Args:
        arg1: Description of arg1.
        arg2: Description of arg2. Defaults to 10.

    Returns:
        Description of return value.

    Raises:
        ValueError: If arg1 is empty.

    Example:
        >>> result = function("test", 5)
        >>> print(result)
        True
    """
```

### Mathematical Notation

Include LaTeX formulas in docstrings for mathematical operations:

```python
"""Compute Earth Mover's Distance.

$$
EMD(P, Q) = \inf_{\gamma \in \Gamma(P, Q)} \int \|x - y\| d\gamma(x, y)
$$
"""
```

---

## Pull Request Process

### Before Submitting

1. **Sync with upstream**:
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

2. **Run all checks**:
   ```bash
   pre-commit run --all-files
   pytest
   mypy src/
   ```

3. **Update documentation** if needed.

4. **Update CHANGELOG.md** with your changes.

### PR Requirements

| Requirement | Description |
|-------------|-------------|
| Title | Clear, descriptive title |
| Description | Detailed explanation of changes |
| Tests | All new code must have tests |
| Documentation | Update relevant docs |
| Changelog | Add entry to CHANGELOG.md |
| CI | All checks must pass |

### Review Process

1. The project lead will review within 48 hours.
2. Address any requested changes.
3. Once approved, the PR will be merged.

---

## Issue Guidelines

### Bug Reports

Include the following information:

- Python version
- PyTorch version
- Operating system
- Minimal reproducible example
- Expected vs actual behavior
- Full error traceback

### Feature Requests

Include:

- Clear description of the feature
- Use case and motivation
- Proposed implementation (if any)
- References to relevant papers

---

## Security Vulnerabilities

Do NOT open public issues for security vulnerabilities. Instead, email the project lead directly at <oyli@dtu.dk>. See [SECURITY.md](SECURITY.md) for details.

---

## Code of Conduct

This project adheres to the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.

---

## License

By contributing to Unbitrium, you agree that your contributions will be licensed under the [European Union Public Licence 1.2 (EUPL-1.2)](LICENSE).

---

## Questions

For questions about contributing, please:

1. Check the [documentation](https://olaflaitinen.github.io/unbitrium/)
2. Search existing [issues](https://github.com/olaflaitinen/unbitrium/issues)
3. Open a new issue if your question is not answered

Thank you for contributing to Unbitrium.
