# Tutorial 133: FL Code Quality

---

## Metadata

| Property | Value |
|----------|-------|
| **Tutorial ID** | 133 |
| **Title** | Federated Learning Code Quality |
| **Category** | Development |
| **Difficulty** | Intermediate |
| **Duration** | 90 minutes |
| **Prerequisites** | Tutorial 001-132 |
| **Author** | Unbitrium Contributors |
| **Last Updated** | January 2026 |

---

## Learning Objectives

By completing this tutorial, you will be able to:

1. **Apply** code quality standards to FL projects
2. **Implement** type hints and documentation
3. **Design** testable FL components
4. **Use** linting and formatting tools
5. **Create** maintainable FL codebases

---

## Prerequisites

Before starting this tutorial, ensure you have:

- Completed tutorials 001-132
- Understanding of Python best practices
- Knowledge of testing frameworks
- Familiarity with type annotations

---

## Background and Theory

### Why Code Quality Matters in FL

Federated learning systems are inherently complex, involving:
- Distributed computation
- Multiple client types
- Communication protocols
- Privacy constraints

High code quality ensures:
- Easier debugging
- Better maintainability
- Fewer production issues
- Faster onboarding

### Code Quality Pillars

```
Code Quality Pillars:
├── Readability
│   ├── Clear naming
│   ├── Consistent style
│   └── Good documentation
├── Reliability
│   ├── Type safety
│   ├── Error handling
│   └── Testing
├── Maintainability
│   ├── Modular design
│   ├── Low coupling
│   └── High cohesion
└── Performance
    ├── Efficient algorithms
    ├── Memory management
    └── Profiling
```

### Python Best Practices for FL

| Practice | Benefit | Tool |
|----------|---------|------|
| Type hints | Catch errors early | mypy |
| Docstrings | Self-documenting | pydocstyle |
| Formatting | Consistent style | black |
| Linting | Bug detection | pylint, ruff |
| Testing | Reliability | pytest |

---

## Implementation Code

```python
#!/usr/bin/env python3
"""
Tutorial 133: Federated Learning Code Quality

This module demonstrates code quality best practices for FL systems,
including type hints, documentation, testing patterns, and clean architecture.

Copyright 2026 Olaf Yunus Laitinen Imanov and Contributors
Released under EUPL 1.2
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Protocol,
    Tuple,
    TypeVar,
    Union,
)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# =============================================================================
# Type Definitions
# =============================================================================

TensorDict = Dict[str, torch.Tensor]
MetricsDict = Dict[str, float]
T = TypeVar("T")


# =============================================================================
# Configuration with Validation
# =============================================================================

@dataclass
class FLConfig:
    """Configuration for federated learning with validation.
    
    This dataclass provides type-safe configuration with built-in
    validation to catch configuration errors early.
    
    Attributes:
        num_rounds: Number of federated learning rounds.
        num_clients: Total number of clients in the system.
        clients_per_round: Clients selected per round.
        input_dim: Dimension of input features.
        hidden_dim: Dimension of hidden layers.
        num_classes: Number of output classes.
        learning_rate: Learning rate for local training.
        batch_size: Batch size for local training.
        local_epochs: Number of local training epochs.
        seed: Random seed for reproducibility.
    
    Raises:
        ValueError: If any configuration parameter is invalid.
    
    Example:
        >>> config = FLConfig(num_rounds=100, num_clients=50)
        >>> print(config.clients_per_round)
        10
    """
    
    num_rounds: int = 50
    num_clients: int = 20
    clients_per_round: int = 10
    input_dim: int = 32
    hidden_dim: int = 64
    num_classes: int = 10
    learning_rate: float = 0.01
    batch_size: int = 32
    local_epochs: int = 3
    seed: int = 42
    
    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        self._validate()
    
    def _validate(self) -> None:
        """Validate all configuration parameters.
        
        Raises:
            ValueError: If any parameter is invalid.
        """
        if self.num_rounds <= 0:
            raise ValueError(f"num_rounds must be positive, got {self.num_rounds}")
        
        if self.num_clients <= 0:
            raise ValueError(f"num_clients must be positive, got {self.num_clients}")
        
        if self.clients_per_round > self.num_clients:
            raise ValueError(
                f"clients_per_round ({self.clients_per_round}) cannot exceed "
                f"num_clients ({self.num_clients})"
            )
        
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {self.learning_rate}")
        
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary.
        
        Returns:
            Dictionary representation of the configuration.
        """
        return {
            "num_rounds": self.num_rounds,
            "num_clients": self.num_clients,
            "clients_per_round": self.clients_per_round,
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "num_classes": self.num_classes,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "local_epochs": self.local_epochs,
            "seed": self.seed,
        }


# =============================================================================
# Protocol Definitions (Interfaces)
# =============================================================================

class ClientProtocol(Protocol):
    """Protocol defining the client interface.
    
    This protocol ensures type safety for client implementations
    without requiring inheritance.
    """
    
    @property
    def client_id(self) -> int:
        """Return the client's unique identifier."""
        ...
    
    def train(self, model: nn.Module) -> Dict[str, Any]:
        """Train on local data and return update."""
        ...


class AggregatorProtocol(Protocol):
    """Protocol defining the aggregator interface."""
    
    def aggregate(
        self,
        updates: List[Dict[str, Any]]
    ) -> TensorDict:
        """Aggregate client updates into global model."""
        ...


class MetricsCollectorProtocol(Protocol):
    """Protocol for metrics collection."""
    
    def record(self, name: str, value: float, step: int) -> None:
        """Record a metric value."""
        ...
    
    def get_history(self, name: str) -> List[float]:
        """Get metric history."""
        ...


# =============================================================================
# Dataset Implementation
# =============================================================================

class FLDataset(Dataset[Tuple[torch.Tensor, torch.Tensor]]):
    """Type-safe dataset implementation for federated learning.
    
    This dataset provides synthetic data for demonstration purposes
    with full type annotations.
    
    Args:
        n: Number of samples.
        dim: Feature dimension.
        num_classes: Number of classes.
        seed: Random seed.
    
    Example:
        >>> dataset = FLDataset(n=100, dim=32, num_classes=10)
        >>> x, y = dataset[0]
        >>> print(x.shape, y.shape)
        torch.Size([32]) torch.Size([])
    """
    
    def __init__(
        self,
        n: int = 100,
        dim: int = 32,
        num_classes: int = 10,
        seed: int = 0
    ) -> None:
        """Initialize the dataset."""
        np.random.seed(seed)
        
        self._x = torch.randn(n, dim, dtype=torch.float32)
        self._y = torch.randint(0, num_classes, (n,), dtype=torch.long)
        
        # Add class-specific patterns
        for i in range(n):
            self._x[i, self._y[i].item() % dim] += 2.0
        
        logger.debug(f"Created dataset with {n} samples")
    
    def __len__(self) -> int:
        """Return the number of samples."""
        return len(self._y)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a sample by index.
        
        Args:
            idx: Sample index.
        
        Returns:
            Tuple of (features, label).
        
        Raises:
            IndexError: If index is out of bounds.
        """
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of bounds for dataset of size {len(self)}")
        
        return self._x[idx], self._y[idx]


# =============================================================================
# Model Implementation
# =============================================================================

class FLModel(nn.Module):
    """Clean, well-documented model implementation.
    
    This model follows PyTorch best practices with clear documentation
    and type annotations.
    
    Args:
        config: Configuration object.
    
    Attributes:
        encoder: Feature encoding layers.
        classifier: Classification head.
    """
    
    def __init__(self, config: FLConfig) -> None:
        """Initialize the model.
        
        Args:
            config: Configuration with model parameters.
        """
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
        )
        
        self.classifier = nn.Linear(config.hidden_dim, config.num_classes)
        
        self._init_weights()
    
    def _init_weights(self) -> None:
        """Initialize model weights using Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim).
        
        Returns:
            Output logits of shape (batch_size, num_classes).
        """
        features = self.encoder(x)
        return self.classifier(features)
    
    def count_parameters(self) -> int:
        """Count trainable parameters.
        
        Returns:
            Number of trainable parameters.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# =============================================================================
# Aggregator Implementation
# =============================================================================

class FedAvgAggregator:
    """FedAvg aggregator with clean implementation.
    
    This aggregator implements weighted averaging of client updates
    with proper type annotations and error handling.
    """
    
    def aggregate(
        self,
        updates: List[Dict[str, Any]],
        weights: Optional[List[float]] = None
    ) -> TensorDict:
        """Aggregate client updates using weighted averaging.
        
        Args:
            updates: List of client updates, each containing 'state_dict'
                and optionally 'num_samples'.
            weights: Optional explicit weights. If None, weights are
                computed from 'num_samples'.
        
        Returns:
            Aggregated state dictionary.
        
        Raises:
            ValueError: If updates list is empty or weights are invalid.
        """
        if not updates:
            raise ValueError("Cannot aggregate empty update list")
        
        # Compute weights
        if weights is None:
            total_samples = sum(u.get("num_samples", 1) for u in updates)
            weights = [u.get("num_samples", 1) / total_samples for u in updates]
        
        # Validate weights
        if len(weights) != len(updates):
            raise ValueError(
                f"Number of weights ({len(weights)}) must match "
                f"number of updates ({len(updates)})"
            )
        
        if abs(sum(weights) - 1.0) > 1e-6:
            logger.warning(f"Weights sum to {sum(weights)}, normalizing")
            total = sum(weights)
            weights = [w / total for w in weights]
        
        # Aggregate
        aggregated: TensorDict = {}
        first_state = updates[0]["state_dict"]
        
        for key in first_state:
            aggregated[key] = sum(
                w * u["state_dict"][key].float()
                for w, u in zip(weights, updates)
            )
        
        logger.debug(f"Aggregated {len(updates)} updates")
        return aggregated


# =============================================================================
# Metrics Collection
# =============================================================================

@dataclass
class MetricsCollector:
    """Collector for training metrics with type safety.
    
    Attributes:
        history: Dictionary mapping metric names to value lists.
    """
    
    history: Dict[str, List[float]] = field(default_factory=dict)
    
    def record(self, name: str, value: float, step: int) -> None:
        """Record a metric value.
        
        Args:
            name: Metric name.
            value: Metric value.
            step: Training step/round.
        """
        if name not in self.history:
            self.history[name] = []
        self.history[name].append(value)
        
        logger.debug(f"Recorded {name}={value:.4f} at step {step}")
    
    def get_history(self, name: str) -> List[float]:
        """Get metric history.
        
        Args:
            name: Metric name.
        
        Returns:
            List of recorded values.
        """
        return self.history.get(name, [])
    
    def get_last(self, name: str, default: float = 0.0) -> float:
        """Get last recorded value.
        
        Args:
            name: Metric name.
            default: Default value if no history.
        
        Returns:
            Last recorded value or default.
        """
        history = self.get_history(name)
        return history[-1] if history else default
    
    def summary(self) -> Dict[str, Dict[str, float]]:
        """Generate summary statistics.
        
        Returns:
            Dictionary with mean, min, max for each metric.
        """
        return {
            name: {
                "mean": float(np.mean(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "last": values[-1]
            }
            for name, values in self.history.items()
            if values
        }


# =============================================================================
# Client Implementation
# =============================================================================

class FLClient:
    """Well-structured FL client implementation.
    
    This client demonstrates clean code practices including
    proper encapsulation and single responsibility.
    
    Args:
        client_id: Unique client identifier.
        dataset: Local training dataset.
        config: Training configuration.
    """
    
    def __init__(
        self,
        client_id: int,
        dataset: FLDataset,
        config: FLConfig
    ) -> None:
        """Initialize the client."""
        self._client_id = client_id
        self._dataset = dataset
        self._config = config
        
        logger.debug(f"Initialized client {client_id}")
    
    @property
    def client_id(self) -> int:
        """Get client identifier."""
        return self._client_id
    
    @property
    def num_samples(self) -> int:
        """Get number of local samples."""
        return len(self._dataset)
    
    def train(self, model: nn.Module) -> Dict[str, Any]:
        """Train on local data.
        
        Args:
            model: Global model to train.
        
        Returns:
            Dictionary containing state_dict, num_samples, and metrics.
        """
        import copy
        
        # Create local copy
        local_model = copy.deepcopy(model)
        optimizer = torch.optim.SGD(
            local_model.parameters(),
            lr=self._config.learning_rate
        )
        
        loader = DataLoader(
            self._dataset,
            batch_size=self._config.batch_size,
            shuffle=True
        )
        
        local_model.train()
        total_loss = 0.0
        num_batches = 0
        
        for _ in range(self._config.local_epochs):
            for x, y in loader:
                optimizer.zero_grad()
                output = local_model(x)
                loss = F.cross_entropy(output, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(local_model.parameters(), 1.0)
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
        
        return {
            "state_dict": {
                k: v.cpu() for k, v in local_model.state_dict().items()
            },
            "num_samples": self.num_samples,
            "avg_loss": total_loss / max(num_batches, 1)
        }


# =============================================================================
# Server Implementation
# =============================================================================

class FLServer:
    """FL server with clean architecture.
    
    The server orchestrates federated learning with proper
    separation of concerns and dependency injection.
    """
    
    def __init__(
        self,
        model: nn.Module,
        clients: List[FLClient],
        test_dataset: FLDataset,
        config: FLConfig,
        aggregator: Optional[FedAvgAggregator] = None,
        metrics: Optional[MetricsCollector] = None
    ) -> None:
        """Initialize the server.
        
        Args:
            model: Global model.
            clients: List of FL clients.
            test_dataset: Test dataset for evaluation.
            config: Training configuration.
            aggregator: Aggregator instance (optional).
            metrics: Metrics collector (optional).
        """
        self.model = model
        self.clients = clients
        self.test_dataset = test_dataset
        self.config = config
        self.aggregator = aggregator or FedAvgAggregator()
        self.metrics = metrics or MetricsCollector()
    
    def evaluate(self) -> MetricsDict:
        """Evaluate model on test data.
        
        Returns:
            Dictionary with accuracy and loss metrics.
        """
        self.model.eval()
        loader = DataLoader(self.test_dataset, batch_size=64)
        
        correct = 0
        total = 0
        total_loss = 0.0
        
        with torch.no_grad():
            for x, y in loader:
                output = self.model(x)
                loss = F.cross_entropy(output, y)
                pred = output.argmax(dim=1)
                
                correct += (pred == y).sum().item()
                total += len(y)
                total_loss += loss.item() * len(y)
        
        return {
            "accuracy": correct / total,
            "loss": total_loss / total
        }
    
    def train_round(self, round_num: int) -> MetricsDict:
        """Execute one federated learning round.
        
        Args:
            round_num: Current round number.
        
        Returns:
            Round metrics.
        """
        # Collect updates
        updates = [c.train(self.model) for c in self.clients]
        
        # Aggregate
        new_state = self.aggregator.aggregate(updates)
        self.model.load_state_dict(new_state)
        
        # Evaluate
        metrics = self.evaluate()
        avg_loss = float(np.mean([u["avg_loss"] for u in updates]))
        
        # Record metrics
        self.metrics.record("accuracy", metrics["accuracy"], round_num)
        self.metrics.record("loss", metrics["loss"], round_num)
        self.metrics.record("train_loss", avg_loss, round_num)
        
        return metrics
    
    def train(self) -> List[MetricsDict]:
        """Run full federated training.
        
        Returns:
            List of metrics for each round.
        """
        logger.info(f"Starting FL training for {self.config.num_rounds} rounds")
        history = []
        
        for round_num in range(self.config.num_rounds):
            metrics = self.train_round(round_num)
            history.append(metrics)
            
            if (round_num + 1) % 10 == 0:
                logger.info(
                    f"Round {round_num + 1}: "
                    f"acc={metrics['accuracy']:.4f}, "
                    f"loss={metrics['loss']:.4f}"
                )
        
        logger.info("Training complete")
        return history


# =============================================================================
# Main Entry Point
# =============================================================================

def main() -> None:
    """Main entry point demonstrating code quality practices."""
    print("=" * 60)
    print("Tutorial 133: FL Code Quality")
    print("=" * 60)
    
    # Configuration with validation
    config = FLConfig()
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    
    # Create components with proper typing
    datasets = [
        FLDataset(seed=i)
        for i in range(config.num_clients)
    ]
    
    clients = [
        FLClient(i, d, config)
        for i, d in enumerate(datasets)
    ]
    
    test_data = FLDataset(n=200, seed=999)
    model = FLModel(config)
    
    logger.info(f"Model parameters: {model.count_parameters():,}")
    
    # Train with clean architecture
    server = FLServer(model, clients, test_data, config)
    history = server.train()
    
    # Print summary
    print("\n" + "=" * 60)
    summary = server.metrics.summary()
    print("Metrics Summary:")
    for name, stats in summary.items():
        print(f"  {name}: mean={stats['mean']:.4f}, last={stats['last']:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
```

---

## Code Quality Checklist

### Before Committing

- [ ] All functions have type hints
- [ ] All public functions have docstrings
- [ ] No linting errors (pylint, ruff)
- [ ] Code is formatted (black)
- [ ] Tests pass (pytest)
- [ ] No hardcoded values

---

## Exercises

1. **Exercise 1**: Add comprehensive unit tests
2. **Exercise 2**: Implement a custom Protocol for models
3. **Exercise 3**: Add structured logging throughout
4. **Exercise 4**: Create property-based tests

---

## References

1. PEP 484 – Type Hints
2. PEP 257 – Docstring Conventions
3. Martin, R.C. (2008). Clean Code
4. Google Python Style Guide

---

*Copyright 2026 Olaf Yunus Laitinen Imanov and Contributors. Released under EUPL 1.2.*
