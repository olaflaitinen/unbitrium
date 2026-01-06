# Benchmarks API Reference

This document provides the API reference for the `unbitrium.bench` module.

---

## Table of Contents

1. [BenchmarkRunner](#benchmarkrunner)
2. [BenchmarkConfig](#benchmarkconfig)
3. [Artifacts](#artifacts)
4. [Reports](#reports)

---

## BenchmarkRunner

```python
from unbitrium.bench import BenchmarkRunner

class BenchmarkRunner:
    """Orchestrates benchmark experiments.

    Args:
        config: Benchmark configuration.
        output_dir: Directory for results.

    Example:
        >>> runner = BenchmarkRunner(config, output_dir="results/")
        >>> results = runner.run()
        >>> runner.generate_report()
    """
```

### Methods

| Method | Description |
|--------|-------------|
| `run()` | Execute benchmark |
| `generate_report()` | Create markdown report |
| `save_artifacts()` | Save results to disk |

---

## BenchmarkConfig

```python
from unbitrium.bench import BenchmarkConfig

@dataclass
class BenchmarkConfig:
    """Benchmark configuration.

    Args:
        name: Experiment name.
        num_clients: Number of clients.
        num_rounds: Training rounds.
        aggregator: Aggregator name.
        partitioner: Partitioner configuration.
    """
```

---

## Artifacts

```python
from unbitrium.bench import Artifacts

class Artifacts:
    """Manages benchmark artifacts.

    Methods:
        save_model(model, path): Save model checkpoint.
        save_metrics(metrics, path): Save metrics JSON.
        save_config(config, path): Save configuration.
    """
```

---

## Reports

```python
from unbitrium.bench import generate_report

def generate_report(
    results: dict,
    output_path: str,
    format: str = "markdown",
) -> None:
    """Generate benchmark report.

    Args:
        results: Benchmark results dictionary.
        output_path: Output file path.
        format: Report format ('markdown', 'html', 'latex').
    """
```

---

*Last updated: January 2026*
