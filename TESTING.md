# Testing Guide

This document describes the testing strategy, infrastructure, and best practices for Unbitrium.

---

## Table of Contents

1. [Overview](#overview)
2. [Test Categories](#test-categories)
3. [Running Tests](#running-tests)
4. [Writing Tests](#writing-tests)
5. [Test Infrastructure](#test-infrastructure)
6. [Coverage Requirements](#coverage-requirements)
7. [Continuous Integration](#continuous-integration)
8. [Debugging Tests](#debugging-tests)

---

## Overview

Unbitrium employs a comprehensive testing strategy to ensure reliability and correctness:

| Metric | Target |
|--------|--------|
| Overall coverage | >= 80% |
| Core modules | >= 90% |
| All tests passing | Required |
| Type checking | mypy --strict |

---

## Test Categories

### Unit Tests

Tests for individual functions and classes in isolation.

| Location | Scope |
|----------|-------|
| `tests/test_aggregators.py` | Aggregation algorithms |
| `tests/test_partitioners.py` | Data partitioning |
| `tests/test_metrics.py` | Heterogeneity metrics |

### Integration Tests

Tests for interactions between components.

| Location | Scope |
|----------|-------|
| `tests/test_integration.py` | End-to-end workflows |

### Smoke Tests

Quick sanity checks for basic functionality.

| Location | Scope |
|----------|-------|
| `tests/test_smoke.py` | Import and basic operations |

---

## Running Tests

### Basic Usage

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific file
pytest tests/test_aggregators.py

# Run specific test
pytest tests/test_aggregators.py::TestFedAvg::test_aggregate_empty_updates
```

### Advanced Options

```bash
# Run with coverage
pytest --cov=src/unbitrium --cov-report=html

# Parallel execution
pytest -n auto

# Stop on first failure
pytest -x

# Show local variables on failure
pytest -l

# Run only fast tests
pytest -m "not slow"

# Run only marked tests
pytest -m integration
```

### Coverage Reports

```bash
# Generate HTML report
pytest --cov=src/unbitrium --cov-report=html

# Open report
# Windows: start htmlcov/index.html
# macOS: open htmlcov/index.html
# Linux: xdg-open htmlcov/index.html
```

---

## Writing Tests

### Test Structure

```python
"""Unit tests for component.

Author: Olaf Yunus Laitinen Imanov <oyli@dtu.dk>
License: EUPL-1.2
"""

from __future__ import annotations

import pytest
import torch

from unbitrium.module import Component


class TestComponent:
    """Tests for Component class."""

    def test_basic_functionality(self) -> None:
        """Test basic use case."""
        component = Component()
        result = component.process(input_data)
        assert result == expected

    def test_edge_case(self) -> None:
        """Test edge case handling."""
        component = Component()
        with pytest.raises(ValueError):
            component.process(invalid_data)
```

### Fixtures

```python
@pytest.fixture
def sample_model() -> torch.nn.Module:
    """Create a simple model for testing."""
    return torch.nn.Linear(10, 5)


@pytest.fixture
def sample_updates(sample_model: torch.nn.Module) -> list[dict]:
    """Create sample client updates."""
    return [
        {"state_dict": sample_model.state_dict(), "num_samples": 100}
        for _ in range(5)
    ]
```

### Parametrized Tests

```python
@pytest.mark.parametrize("num_clients,alpha,expected_variance", [
    (10, 0.1, "high"),
    (10, 1.0, "medium"),
    (10, 10.0, "low"),
])
def test_dirichlet_variance(
    self,
    num_clients: int,
    alpha: float,
    expected_variance: str,
) -> None:
    """Test Dirichlet partitioner with various alpha values."""
    partitioner = DirichletPartitioner(num_clients=num_clients, alpha=alpha)
    # Test implementation
```

### Markers

```python
@pytest.mark.slow
def test_large_scale_simulation(self) -> None:
    """Test with 1000 clients (slow)."""
    pass


@pytest.mark.gpu
def test_cuda_acceleration(self) -> None:
    """Test GPU functionality."""
    pass


@pytest.mark.integration
def test_end_to_end_pipeline(self) -> None:
    """Test complete FL pipeline."""
    pass
```

---

## Test Infrastructure

### Conftest

Shared fixtures in `tests/conftest.py`:

```python
"""Shared test fixtures.

Author: Olaf Yunus Laitinen Imanov <oyli@dtu.dk>
License: EUPL-1.2
"""

import pytest
import numpy as np
import torch


@pytest.fixture(autouse=True)
def set_random_seed() -> None:
    """Ensure deterministic tests."""
    np.random.seed(42)
    torch.manual_seed(42)


@pytest.fixture
def simple_model() -> torch.nn.Module:
    """Simple model for testing."""
    return torch.nn.Sequential(
        torch.nn.Linear(10, 5),
        torch.nn.ReLU(),
        torch.nn.Linear(5, 2),
    )
```

### pytest.ini Configuration

```ini
[pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
markers =
    slow: marks tests as slow
    gpu: marks tests requiring GPU
    integration: marks integration tests
addopts = --strict-markers
```

---

## Coverage Requirements

### Minimum Coverage by Module

| Module | Minimum |
|--------|---------|
| `aggregators/` | 90% |
| `partitioning/` | 85% |
| `metrics/` | 85% |
| `core/` | 80% |
| `privacy/` | 80% |
| `simulation/` | 75% |
| `systems/` | 75% |
| `bench/` | 70% |

### Excluding from Coverage

```python
# pragma: no cover - for intentionally untested code
if TYPE_CHECKING:  # pragma: no cover
    from typing import TypeAlias
```

---

## Continuous Integration

### GitHub Actions

Tests run automatically on:

| Event | Tests |
|-------|-------|
| Push to main | Full suite |
| Pull request | Full suite |
| Nightly | Full + slow tests |

### CI Configuration

See `.github/workflows/ci.yml` for configuration.

### Pre-commit Hooks

```bash
# Install hooks
pre-commit install

# Run manually
pre-commit run --all-files
```

---

## Debugging Tests

### Verbose Output

```bash
# Show print statements
pytest -s

# Show local variables
pytest -l

# Drop into debugger on failure
pytest --pdb
```

### Using pdb

```python
def test_complex_logic(self) -> None:
    """Debug complex test."""
    import pdb; pdb.set_trace()
    # Execution pauses here
    result = complex_function()
    assert result == expected
```

### Logging

```python
import logging

def test_with_logging(caplog) -> None:
    """Capture log output."""
    with caplog.at_level(logging.DEBUG):
        function_that_logs()
    assert "expected message" in caplog.text
```

---

## Best Practices

### Do

- Write tests before or alongside code
- Use descriptive test names
- Test edge cases and error conditions
- Keep tests independent and isolated
- Use fixtures for shared setup
- Run tests locally before pushing

### Don't

- Skip tests without justification
- Use random data without seeding
- Test implementation details
- Create tests that depend on order
- Ignore flaky tests

---

*Last updated: January 2026*
