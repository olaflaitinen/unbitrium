# Style Guide

This document defines the coding and documentation style standards for Unbitrium.

---

## Table of Contents

1. [Python Style](#python-style)
2. [Naming Conventions](#naming-conventions)
3. [Documentation Style](#documentation-style)
4. [File Organization](#file-organization)
5. [Formatting Tools](#formatting-tools)
6. [Type Hints](#type-hints)
7. [Imports](#imports)
8. [Comments](#comments)

---

## Python Style

Unbitrium follows the [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html) with the following specifications:

### Line Length

| Context | Limit |
|---------|-------|
| Code | 88 characters |
| Docstrings | 79 characters |
| Comments | 79 characters |

### Indentation

- Use 4 spaces for indentation
- Never use tabs
- Continuation lines should align with opening delimiter or use hanging indent

```python
# Aligned with opening delimiter
def function(arg1: str,
             arg2: int,
             arg3: float) -> bool:
    pass

# Hanging indent
def function(
    arg1: str,
    arg2: int,
    arg3: float,
) -> bool:
    pass
```

### Strings

- Use double quotes for strings
- Use triple double quotes for docstrings
- Use f-strings for interpolation

```python
message = "Hello, World!"
name = "Unbitrium"
greeting = f"Welcome to {name}"
```

---

## Naming Conventions

### General Rules

| Type | Convention | Example |
|------|------------|---------|
| Module | snake_case | `federated_avg.py` |
| Class | PascalCase | `FederatedAverage` |
| Function | snake_case | `compute_gradient_variance` |
| Method | snake_case | `aggregate_updates` |
| Variable | snake_case | `client_updates` |
| Constant | UPPER_SNAKE_CASE | `DEFAULT_LEARNING_RATE` |
| Type variable | PascalCase | `TensorType` |

### Specific Conventions

```python
# Classes - PascalCase
class DirichletPartitioner:
    pass

# Functions - snake_case, verb-noun
def compute_label_entropy(labels: np.ndarray) -> float:
    pass

# Private methods - leading underscore
def _validate_inputs(self) -> None:
    pass

# Constants - UPPER_SNAKE_CASE
DEFAULT_EPSILON = 1.0
MAX_GRADIENT_NORM = 1.0

# Type aliases - PascalCase
StateDict = dict[str, torch.Tensor]
ClientUpdate = dict[str, Any]
```

### Abbreviations

| Full Name | Abbreviation |
|-----------|--------------|
| Federated Learning | FL |
| Differential Privacy | DP |
| Stochastic Gradient Descent | SGD |
| Non-Independent and Identically Distributed | non-IID |

---

## Documentation Style

### Module Docstrings

Every module must have a docstring:

```python
"""Short description of module.

Extended description if needed. Can span multiple lines
and provide context about the module's purpose.

Author: Olaf Yunus Laitinen Imanov <oyli@dtu.dk>
License: EUPL-1.2
"""

from __future__ import annotations
```

### Class Docstrings

```python
class FedAvg(Aggregator):
    """Federated Averaging aggregation algorithm.

    Implements the FedAvg algorithm from McMahan et al. (2017).
    Aggregates client updates using weighted averaging based
    on the number of samples.

    Mathematical formulation:

    $$
    w^{t+1} = \sum_{k=1}^K \frac{n_k}{n} w_k^t
    $$

    Args:
        momentum: Optional momentum coefficient.

    Example:
        >>> aggregator = FedAvg(momentum=0.9)
        >>> new_model, metrics = aggregator.aggregate(updates, model)
    """
```

### Function Docstrings

```python
def compute_emd(
    labels: np.ndarray,
    client_indices: dict[int, list[int]],
) -> float:
    """Compute Earth Mover's Distance from global distribution.

    Calculates the average EMD between each client's label
    distribution and the global distribution.

    Args:
        labels: Array of class labels for all samples.
        client_indices: Mapping from client ID to sample indices.

    Returns:
        Average EMD across all clients.

    Raises:
        ValueError: If labels is empty.

    Example:
        >>> labels = np.array([0, 1, 0, 1, 2])
        >>> indices = {0: [0, 1], 1: [2, 3, 4]}
        >>> emd = compute_emd(labels, indices)
    """
```

---

## File Organization

### Standard Module Layout

```python
"""Module docstring."""

from __future__ import annotations

# Standard library imports
import os
import sys
from typing import Any

# Third-party imports
import numpy as np
import torch

# Local imports
from unbitrium.core import utils


# Constants
DEFAULT_VALUE = 1.0


# Classes
class MainClass:
    """Main class docstring."""
    pass


# Functions
def main_function() -> None:
    """Main function docstring."""
    pass


# Main guard
if __name__ == "__main__":
    main_function()
```

### Import Order

1. `from __future__ import annotations`
2. Standard library imports
3. Blank line
4. Third-party imports
5. Blank line
6. Local imports

---

## Formatting Tools

### Black

Code formatting:

```bash
black src/ tests/
```

Configuration in `pyproject.toml`:

```toml
[tool.black]
line-length = 88
target-version = ["py310"]
```

### isort

Import sorting:

```bash
isort src/ tests/
```

Configuration in `pyproject.toml`:

```toml
[tool.isort]
profile = "black"
line_length = 88
```

### Ruff

Linting:

```bash
ruff check src/ tests/
```

---

## Type Hints

### Required Type Hints

All functions must have complete type hints:

```python
def aggregate(
    self,
    updates: list[dict[str, Any]],
    current_global_model: torch.nn.Module,
) -> tuple[torch.nn.Module, dict[str, float]]:
    """Aggregate client updates."""
    pass
```

### Type Annotations

```python
from __future__ import annotations

from typing import Any, Callable, TypeVar

# Type aliases
StateDict = dict[str, torch.Tensor]
Metric = Callable[[np.ndarray, dict[int, list[int]]], float]

# Type variables
T = TypeVar("T", bound=torch.nn.Module)
```

### Optional Types

```python
from __future__ import annotations

def process(
    data: np.ndarray,
    threshold: float | None = None,
) -> list[int]:
    """Process with optional threshold."""
    pass
```

---

## Imports

### Absolute vs Relative

Use absolute imports:

```python
# Good
from unbitrium.aggregators import FedAvg
from unbitrium.metrics.heterogeneity import compute_emd

# Avoid
from .fedavg import FedAvg
from ..metrics import compute_emd
```

### Import Style

```python
# Import specific names
from typing import Any, Dict, List

# Not entire modules when possible
# Avoid: import typing

# Exception: numpy and torch
import numpy as np
import torch
import torch.nn as nn
```

---

## Comments

### Inline Comments

```python
# Good: Explain why, not what
weight = 1.0 / (1.0 + distance)  # Inverse distance weighting

# Avoid: Stating the obvious
x = x + 1  # Increment x by 1
```

### TODO Comments

```python
# TODO(oyli): Implement momentum-based aggregation
# TODO: Add GPU support for large-scale experiments
```

### Block Comments

```python
# ====================
# Section: Aggregation
# ====================
```

---

*Last updated: January 2026*
