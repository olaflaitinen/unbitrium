# Development Guide

This document provides comprehensive guidelines for developing and extending Unbitrium.

---

## Table of Contents

1. [Development Setup](#development-setup)
2. [Project Structure](#project-structure)
3. [Development Workflow](#development-workflow)
4. [Adding New Features](#adding-new-features)
5. [Testing](#testing)
6. [Documentation](#documentation)
7. [Performance Optimization](#performance-optimization)
8. [Debugging](#debugging)
9. [Release Process](#release-process)

---

## Development Setup

### Prerequisites

| Requirement | Version | Purpose |
|-------------|---------|---------|
| Python | >= 3.10 | Runtime |
| Git | >= 2.30 | Version control |
| PyTorch | >= 2.0 | Deep learning |
| Make | Any | Build automation |

### Installation

```bash
# Clone the repository
git clone https://github.com/olaflaitinen/unbitrium.git
cd unbitrium

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On Windows:
.venv\Scripts\activate
# On Unix/macOS:
source .venv/bin/activate

# Install in development mode
pip install -e ".[dev,docs]"

# Install pre-commit hooks
pre-commit install
```

### Verify Installation

```bash
# Run tests
pytest

# Type checking
mypy src/

# Linting
ruff check src/

# Import the library
python -c "import unbitrium; print(unbitrium.__version__)"
```

---

## Project Structure

```
unbitrium/
    .github/                  # GitHub configuration
        workflows/            # CI/CD workflows
    assets/                   # Static assets
    benchmarks/               # Benchmark scripts
        configs/              # Benchmark configurations
    docs/                     # Documentation
        api/                  # API reference
        research/             # Research notes
        tutorials/            # Tutorial files
        validation/           # Validation reports
    examples/                 # Example scripts
    src/
        unbitrium/            # Main package
            aggregators/      # Aggregation algorithms
            bench/            # Benchmark utilities
            core/             # Core simulation infrastructure
            datasets/         # Dataset loaders
            metrics/          # Heterogeneity metrics
            partitioning/     # Data partitioning
            privacy/          # Privacy mechanisms
            simulation/       # Client/server simulation
            systems/          # Device/energy models
    tests/                    # Test suite
```

### Key Files

| File | Purpose |
|------|---------|
| `pyproject.toml` | Package configuration |
| `mkdocs.yml` | Documentation configuration |
| `.pre-commit-config.yaml` | Pre-commit hooks |
| `CONTRIBUTING.md` | Contribution guidelines |

---

## Development Workflow

### Branch Strategy

| Branch | Purpose |
|--------|---------|
| `main` | Stable release branch |
| `develop` | Integration branch |
| `feature/*` | Feature development |
| `bugfix/*` | Bug fixes |
| `release/*` | Release preparation |

### Creating a Feature Branch

```bash
# Sync with upstream
git fetch upstream
git checkout main
git rebase upstream/main

# Create feature branch
git checkout -b feature/my-feature
```

### Making Changes

1. **Write code** following style guidelines
2. **Write tests** covering new functionality
3. **Update documentation** as needed
4. **Run checks** before committing

```bash
# Format code
black src/ tests/
isort src/ tests/

# Lint
ruff check src/ tests/

# Type check
mypy src/

# Run tests
pytest
```

### Committing

Follow conventional commit format:

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

Types: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`

Examples:
```
feat(aggregators): add FedNova aggregation algorithm
fix(partitioning): correct edge case in Dirichlet sampling
docs(tutorials): add tutorial for custom metrics
```

---

## Adding New Features

### Adding a New Aggregator

1. **Create module** in `src/unbitrium/aggregators/`:

```python
"""NewAggregator implementation.

Description of the algorithm.

Author: Your Name <email@example.com>
License: EUPL-1.2
"""

from __future__ import annotations

from typing import Any

import torch

from unbitrium.aggregators.base import Aggregator


class NewAggregator(Aggregator):
    """Description of aggregator.

    Args:
        param1: Description.

    Example:
        >>> agg = NewAggregator(param1=0.1)
    """

    def __init__(self, param1: float = 0.1) -> None:
        """Initialize aggregator."""
        self.param1 = param1

    def aggregate(
        self,
        updates: list[dict[str, Any]],
        current_global_model: torch.nn.Module,
    ) -> tuple[torch.nn.Module, dict[str, float]]:
        """Aggregate client updates.

        Args:
            updates: Client updates.
            current_global_model: Current model.

        Returns:
            Updated model and metrics.
        """
        # Implementation
        pass
```

2. **Export** in `src/unbitrium/aggregators/__init__.py`

3. **Add tests** in `tests/test_aggregators.py`

4. **Document** in `docs/api/aggregators.md`

### Adding a New Metric

1. **Create function** in `src/unbitrium/metrics/`:

```python
def compute_new_metric(
    labels: np.ndarray,
    client_indices: dict[int, list[int]],
) -> float:
    """Compute new metric.

    Args:
        labels: Class labels.
        client_indices: Client data indices.

    Returns:
        Metric value.
    """
    # Implementation
    pass
```

2. **Export** in `src/unbitrium/metrics/__init__.py`

3. **Add tests** in `tests/test_metrics.py`

---

## Testing

### Running Tests

```bash
# All tests
pytest

# Specific file
pytest tests/test_aggregators.py

# Specific test
pytest tests/test_aggregators.py::TestFedAvg::test_aggregate_empty

# With coverage
pytest --cov=src/unbitrium --cov-report=html

# Parallel execution
pytest -n auto
```

### Writing Tests

```python
class TestNewFeature:
    """Tests for new feature."""

    def test_basic_functionality(self) -> None:
        """Test basic use case."""
        result = new_function(input)
        assert result == expected

    @pytest.mark.parametrize("input,expected", [
        (1, 1),
        (2, 4),
        (3, 9),
    ])
    def test_parametrized(self, input: int, expected: int) -> None:
        """Test with various inputs."""
        assert square(input) == expected

    @pytest.mark.slow
    def test_expensive_operation(self) -> None:
        """Test that requires significant computation."""
        # Long-running test
```

### Test Markers

| Marker | Purpose |
|--------|---------|
| `@pytest.mark.slow` | Long-running tests |
| `@pytest.mark.gpu` | GPU-required tests |
| `@pytest.mark.integration` | Integration tests |

---

## Documentation

### Building Documentation

```bash
# Build docs
mkdocs build

# Serve locally
mkdocs serve

# Open http://localhost:8000
```

### Writing Documentation

Follow these guidelines:

1. **Use Google-style docstrings**
2. **Include mathematical notation** in LaTeX
3. **Provide examples** for all public APIs
4. **Cross-reference** related components

---

## Performance Optimization

### Profiling

```python
import cProfile
import pstats

with cProfile.Profile() as pr:
    # Code to profile
    result = expensive_function()

stats = pstats.Stats(pr)
stats.sort_stats('cumulative')
stats.print_stats(10)
```

### GPU Memory

```python
import torch

# Check memory usage
print(f"Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
print(f"Cached: {torch.cuda.memory_reserved() / 1e9:.2f} GB")

# Clear cache
torch.cuda.empty_cache()
```

---

## Debugging

### Debug Mode

```python
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("unbitrium")
```

### Common Issues

| Issue | Solution |
|-------|----------|
| Import errors | Check installation: `pip install -e "."` |
| Type errors | Run mypy: `mypy src/` |
| Test failures | Run specific test with `-v` flag |

---

## Release Process

### Version Bumping

1. Update `src/unbitrium/__init__.py`
2. Update `pyproject.toml`
3. Update `CHANGELOG.md`

### Release Checklist

- [ ] All tests pass
- [ ] Documentation updated
- [ ] CHANGELOG updated
- [ ] Version bumped
- [ ] Tag created
- [ ] Package published

---

*Last updated: January 2026*
