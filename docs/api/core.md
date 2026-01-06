# Core API Reference

This document provides the API reference for the `unbitrium.core` module.

---

## Table of Contents

1. [SimulationEngine](#simulationengine)
2. [EventSystem](#eventsystem)
3. [ProvenanceTracker](#provenancetracker)
4. [RNGManager](#rngmanager)
5. [Utilities](#utilities)

---

## SimulationEngine

```python
from unbitrium.core import SimulationEngine
```

### Overview

The `SimulationEngine` orchestrates federated learning simulations, managing client selection, training coordination, and aggregation.

### Class Definition

```python
class SimulationEngine:
    """Orchestrates federated learning simulations.

    Args:
        config: Configuration dictionary.
        aggregator: Aggregation algorithm.
        model: Global model.
        client_datasets: Dictionary of client datasets.
        mode: Execution mode ('sync' or 'async').

    Example:
        >>> engine = SimulationEngine(
        ...     config=config,
        ...     aggregator=FedAvg(),
        ...     model=global_model,
        ...     client_datasets=datasets,
        ... )
        >>> results = engine.run(num_rounds=10)
    """
```

### Methods

| Method | Description |
|--------|-------------|
| `run(num_rounds)` | Execute simulation for specified rounds |
| `select_clients(num_clients)` | Select clients for participation |
| `train_round()` | Execute single training round |
| `aggregate(updates)` | Aggregate client updates |

---

## EventSystem

```python
from unbitrium.core import EventSystem
```

### Overview

Publish-subscribe event system for simulation lifecycle events.

### Class Definition

```python
class EventSystem:
    """Event management system.

    Example:
        >>> events = EventSystem()
        >>> events.subscribe("round_end", callback_fn)
        >>> events.publish("round_end", round_num=5)
    """
```

### Methods

| Method | Description |
|--------|-------------|
| `subscribe(event, callback)` | Register callback for event |
| `unsubscribe(event, callback)` | Remove callback |
| `publish(event, **kwargs)` | Trigger event with data |
| `clear(event)` | Clear all callbacks for event |

### Events

| Event | Data |
|-------|------|
| `round_start` | `round_num` |
| `round_end` | `round_num`, `metrics` |
| `client_selected` | `client_ids` |
| `aggregation_complete` | `metrics` |

---

## ProvenanceTracker

```python
from unbitrium.core import ProvenanceTracker
```

### Overview

Tracks experiment metadata for reproducibility.

### Class Definition

```python
class ProvenanceTracker:
    """Experiment provenance tracking.

    Args:
        experiment_name: Name of the experiment.
        output_dir: Directory for provenance files.

    Example:
        >>> tracker = ProvenanceTracker("exp_001")
        >>> tracker.log_config(config)
        >>> tracker.log_metric("accuracy", 0.95)
        >>> tracker.save_manifest()
    """
```

### Methods

| Method | Description |
|--------|-------------|
| `log_config(config)` | Log configuration |
| `log_metric(name, value)` | Log metric value |
| `log_artifact(path)` | Log artifact path |
| `save_manifest()` | Save provenance manifest |
| `load_manifest(path)` | Load existing manifest |

---

## RNGManager

```python
from unbitrium.core import RNGManager
```

### Overview

Manages random number generators for reproducibility.

### Class Definition

```python
class RNGManager:
    """Deterministic random number management.

    Args:
        seed: Master seed for all generators.

    Example:
        >>> rng = RNGManager(seed=42)
        >>> rng.set_all_seeds()
        >>> child_rng = rng.fork(client_id=5)
    """
```

### Methods

| Method | Description |
|--------|-------------|
| `set_all_seeds()` | Set seeds for numpy, torch, random |
| `fork(client_id)` | Create child RNG for client |
| `get_state()` | Get RNG state |
| `set_state(state)` | Restore RNG state |
| `numpy_generator()` | Get numpy Generator |
| `torch_generator()` | Get torch Generator |

---

## Utilities

```python
from unbitrium.core.utils import setup_logging, set_seed, create_provenance
```

### Functions

| Function | Description |
|----------|-------------|
| `setup_logging(level)` | Configure logging |
| `set_seed(seed)` | Set all random seeds |
| `create_provenance(name)` | Create provenance tracker |

### Example

```python
from unbitrium.core.utils import setup_logging, set_seed

# Setup
setup_logging("INFO")
set_seed(42)

# Your experiment code
```

---

*Last updated: January 2026*
