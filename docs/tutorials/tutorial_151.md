# Tutorial 151: FL Best Practices

---

## Metadata

| Property | Value |
|----------|-------|
| **Tutorial ID** | 151 |
| **Title** | Federated Learning Best Practices |
| **Category** | Guidelines |
| **Difficulty** | Intermediate |
| **Duration** | 90 minutes |
| **Prerequisites** | Tutorial 001-150 |
| **Author** | Unbitrium Contributors |
| **Last Updated** | January 2026 |

---

## Learning Objectives

By completing this tutorial, you will be able to:

1. **Apply** industry best practices for FL development
2. **Design** robust and scalable FL systems
3. **Implement** production-ready FL code
4. **Avoid** common pitfalls and anti-patterns
5. **Optimize** FL systems for real-world deployment

---

## Prerequisites

Before starting this tutorial, ensure you have:

- Completed tutorials 001-150
- Experience with FL implementation
- Understanding of distributed systems
- Familiarity with ML production practices

---

## Background and Theory

### Why Best Practices Matter

FL systems are complex distributed systems that must:
- Handle heterogeneous client behavior
- Maintain privacy guarantees
- Scale to thousands of clients
- Recover from failures gracefully

Following best practices ensures:
- Reliable training convergence
- Maintainable codebase
- Efficient resource usage
- Production readiness

### Best Practice Categories

```
FL Best Practices:
├── Architecture
│   ├── Modular design
│   ├── Clear interfaces
│   └── Separation of concerns
├── Data Handling
│   ├── Validation
│   ├── Preprocessing
│   └── Privacy protection
├── Training
│   ├── Hyperparameter tuning
│   ├── Convergence monitoring
│   └── Gradient handling
├── Communication
│   ├── Compression
│   ├── Retries
│   └── Batching
└── Operations
    ├── Logging
    ├── Monitoring
    └── Versioning
```

### Principle Summary

| Principle | Description | Priority |
|-----------|-------------|----------|
| Deep Copy | Always clone models | Critical |
| Weighted Aggregation | Weight by samples | Critical |
| Gradient Clipping | Prevent explosions | High |
| Data Validation | Check inputs | High |
| Error Handling | Graceful failures | High |

---

## Implementation Code

```python
#!/usr/bin/env python3
"""
Tutorial 151: Federated Learning Best Practices

This module demonstrates production-quality FL code following
industry best practices.

Copyright 2026 Olaf Yunus Laitinen Imanov and Contributors
Released under EUPL 1.2
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Protocol
import copy
import logging
from abc import ABC, abstractmethod
from contextlib import contextmanager
import traceback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# =============================================================================
# Best Practice 1: Type-Safe Configuration with Validation
# =============================================================================

@dataclass
class BestPracticeConfig:
    """Configuration with comprehensive validation.

    Best Practice: Always validate configuration at initialization.
    This catches errors early before training starts.
    """

    # FL parameters
    num_rounds: int = 50
    num_clients: int = 10
    clients_per_round: int = 5
    min_clients_per_round: int = 2  # Minimum for valid aggregation

    # Model parameters
    input_dim: int = 32
    hidden_dim: int = 64
    num_classes: int = 10

    # Training parameters
    learning_rate: float = 0.01
    batch_size: int = 32
    local_epochs: int = 3
    gradient_clip_norm: float = 1.0

    # Robustness
    max_retries: int = 3
    timeout_seconds: float = 300.0

    seed: int = 42

    def __post_init__(self) -> None:
        """Validate all configuration parameters."""
        self._validate()
        logger.info("Configuration validated successfully")

    def _validate(self) -> None:
        """Comprehensive validation."""
        errors = []

        # Positive checks
        if self.num_rounds <= 0:
            errors.append(f"num_rounds must be positive: {self.num_rounds}")
        if self.num_clients <= 0:
            errors.append(f"num_clients must be positive: {self.num_clients}")
        if self.clients_per_round <= 0:
            errors.append(f"clients_per_round must be positive: {self.clients_per_round}")

        # Consistency checks
        if self.clients_per_round > self.num_clients:
            errors.append(
                f"clients_per_round ({self.clients_per_round}) cannot exceed "
                f"num_clients ({self.num_clients})"
            )
        if self.min_clients_per_round > self.clients_per_round:
            errors.append(
                f"min_clients_per_round ({self.min_clients_per_round}) cannot exceed "
                f"clients_per_round ({self.clients_per_round})"
            )

        # Range checks
        if not (0 < self.learning_rate <= 1):
            errors.append(f"learning_rate unusual: {self.learning_rate}")
        if self.gradient_clip_norm <= 0:
            errors.append(f"gradient_clip_norm must be positive: {self.gradient_clip_norm}")

        if errors:
            for error in errors:
                logger.error(f"Config error: {error}")
            raise ValueError(f"Configuration errors: {errors}")


# =============================================================================
# Best Practice 2: Protocol-Based Interfaces
# =============================================================================

class ClientProtocol(Protocol):
    """Protocol defining expected client interface.

    Best Practice: Use protocols for loose coupling and easier testing.
    """

    @property
    def client_id(self) -> int:
        """Unique client identifier."""
        ...

    def train(self, model: nn.Module) -> Dict[str, Any]:
        """Train on local data."""
        ...


class AggregatorProtocol(Protocol):
    """Protocol for aggregation strategies."""

    def aggregate(self, updates: List[Dict]) -> Dict[str, torch.Tensor]:
        """Aggregate client updates."""
        ...


# =============================================================================
# Best Practice 3: Clean Dataset Implementation
# =============================================================================

class ValidatedDataset(Dataset):
    """Dataset with built-in validation.

    Best Practice: Validate data at creation, not at training time.
    """

    def __init__(
        self,
        n: int = 100,
        dim: int = 32,
        classes: int = 10,
        seed: int = 0
    ):
        if n <= 0:
            raise ValueError(f"n must be positive: {n}")
        if dim <= 0:
            raise ValueError(f"dim must be positive: {dim}")
        if classes <= 0:
            raise ValueError(f"classes must be positive: {classes}")

        np.random.seed(seed)

        self.x = torch.randn(n, dim, dtype=torch.float32)
        self.y = torch.randint(0, classes, (n,), dtype=torch.long)

        # Add class patterns
        for i in range(n):
            self.x[i, self.y[i].item() % dim] += 2.0

        # Validate generated data
        self._validate_data()

    def _validate_data(self) -> None:
        """Validate data integrity."""
        if torch.isnan(self.x).any():
            raise ValueError("NaN values in features")
        if torch.isinf(self.x).any():
            raise ValueError("Inf values in features")
        if (self.y < 0).any() or (self.y >= 10).any():
            raise ValueError("Labels out of range")

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of bounds")
        return self.x[idx], self.y[idx]


# =============================================================================
# Best Practice 4: Robust Model Implementation
# =============================================================================

class RobustModel(nn.Module):
    """Model with initialization and forward pass best practices.

    Best Practice: Use proper weight initialization and
    include checks in forward pass.
    """

    def __init__(self, config: BestPracticeConfig):
        super().__init__()

        self.config = config

        self.layers = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.num_classes)
        )

        # Best Practice: Explicit weight initialization
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights using Xavier/Glorot initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with optional validation."""
        # Best Practice: Validate input in debug mode
        if __debug__:
            if torch.isnan(x).any():
                raise ValueError("NaN in input")
            if torch.isinf(x).any():
                raise ValueError("Inf in input")

        output = self.layers(x)

        # Best Practice: Validate output in debug mode
        if __debug__:
            if torch.isnan(output).any():
                raise ValueError("NaN in output")

        return output

    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# =============================================================================
# Best Practice 5: Proper Client Implementation
# =============================================================================

class BestPracticeClient:
    """Client following all best practices.

    Key Best Practices:
    1. Deep copy model before training
    2. Use gradient clipping
    3. Handle errors gracefully
    4. Return comprehensive metrics
    """

    def __init__(
        self,
        client_id: int,
        dataset: ValidatedDataset,
        config: BestPracticeConfig
    ):
        self._client_id = client_id
        self._dataset = dataset
        self._config = config

    @property
    def client_id(self) -> int:
        return self._client_id

    def train(self, model: nn.Module) -> Dict[str, Any]:
        """Train with all best practices applied."""
        try:
            return self._train_internal(model)
        except Exception as e:
            logger.error(f"Client {self.client_id} training failed: {e}")
            logger.debug(traceback.format_exc())
            return self._create_error_result(str(e))

    def _train_internal(self, model: nn.Module) -> Dict[str, Any]:
        """Internal training implementation."""

        # Best Practice 1: Deep copy model
        local_model = copy.deepcopy(model)

        # Best Practice 2: Use appropriate optimizer
        optimizer = torch.optim.SGD(
            local_model.parameters(),
            lr=self._config.learning_rate,
            momentum=0.9  # Often helps convergence
        )

        loader = DataLoader(
            self._dataset,
            batch_size=self._config.batch_size,
            shuffle=True,  # Best Practice: Shuffle training data
            drop_last=False
        )

        local_model.train()
        losses = []

        for epoch in range(self._config.local_epochs):
            epoch_loss = 0.0
            num_batches = 0

            for x, y in loader:
                # Best Practice 3: Zero gradients before backward
                optimizer.zero_grad()

                output = local_model(x)
                loss = F.cross_entropy(output, y)

                loss.backward()

                # Best Practice 4: Clip gradients
                torch.nn.utils.clip_grad_norm_(
                    local_model.parameters(),
                    self._config.gradient_clip_norm
                )

                optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

            losses.append(epoch_loss / num_batches)

        # Best Practice 5: Return comprehensive results
        return {
            "state_dict": {
                k: v.cpu() for k, v in local_model.state_dict().items()
            },
            "num_samples": len(self._dataset),
            "avg_loss": np.mean(losses),
            "final_loss": losses[-1],
            "client_id": self.client_id,
            "status": "success"
        }

    def _create_error_result(self, error: str) -> Dict[str, Any]:
        """Create result for failed training."""
        return {
            "status": "error",
            "error": error,
            "client_id": self.client_id
        }


# =============================================================================
# Best Practice 6: Robust Server Implementation
# =============================================================================

class BestPracticeServer:
    """Server following all best practices."""

    def __init__(
        self,
        model: nn.Module,
        clients: List[BestPracticeClient],
        test_data: ValidatedDataset,
        config: BestPracticeConfig
    ):
        self.model = model
        self.clients = clients
        self.test_data = test_data
        self.config = config

        self.history: List[Dict] = []
        self.failed_rounds: List[int] = []

    def select_clients(self) -> List[BestPracticeClient]:
        """Select clients for current round."""
        n = min(self.config.clients_per_round, len(self.clients))
        indices = np.random.choice(len(self.clients), n, replace=False)
        return [self.clients[i] for i in indices]

    def aggregate(self, updates: List[Dict]) -> bool:
        """Aggregate with validation.

        Best Practice: Validate updates before aggregation.
        """
        # Filter successful updates
        successful = [u for u in updates if u.get("status") == "success"]

        if len(successful) < self.config.min_clients_per_round:
            logger.error(
                f"Not enough successful clients: "
                f"{len(successful)} < {self.config.min_clients_per_round}"
            )
            return False

        # Best Practice: Weighted aggregation
        total_samples = sum(u["num_samples"] for u in successful)
        new_state = {}

        for key in successful[0]["state_dict"]:
            weighted_sum = sum(
                (u["num_samples"] / total_samples) * u["state_dict"][key].float()
                for u in successful
            )

            # Best Practice: Validate aggregated weights
            if torch.isnan(weighted_sum).any():
                logger.error(f"NaN in aggregated weights for {key}")
                return False

            new_state[key] = weighted_sum

        self.model.load_state_dict(new_state)
        return True

    def evaluate(self) -> Dict[str, float]:
        """Evaluate model."""
        self.model.eval()  # Best Practice: Set to eval mode
        loader = DataLoader(self.test_data, batch_size=64)

        correct = 0
        total = 0
        total_loss = 0.0

        with torch.no_grad():  # Best Practice: No gradients for evaluation
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

    def train(self) -> List[Dict]:
        """Run FL training."""
        logger.info(f"Starting FL training for {self.config.num_rounds} rounds")

        for round_num in range(self.config.num_rounds):
            # Select clients
            selected = self.select_clients()

            # Collect updates
            updates = [c.train(self.model) for c in selected]

            # Aggregate
            success = self.aggregate(updates)

            if not success:
                self.failed_rounds.append(round_num)
                logger.warning(f"Round {round_num} aggregation failed")
                continue

            # Evaluate
            metrics = self.evaluate()

            # Track
            successful_count = len([u for u in updates if u.get("status") == "success"])
            record = {
                "round": round_num,
                **metrics,
                "successful_clients": successful_count
            }
            self.history.append(record)

            if (round_num + 1) % 10 == 0:
                logger.info(
                    f"Round {round_num + 1}: "
                    f"acc={metrics['accuracy']:.4f}, "
                    f"loss={metrics['loss']:.4f}"
                )

        return self.history


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    """Main entry point."""
    print("=" * 60)
    print("Tutorial 151: FL Best Practices")
    print("=" * 60)

    # Configuration
    config = BestPracticeConfig()
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    # Create components
    datasets = [
        ValidatedDataset(seed=i)
        for i in range(config.num_clients)
    ]

    clients = [
        BestPracticeClient(i, d, config)
        for i, d in enumerate(datasets)
    ]

    test_data = ValidatedDataset(n=200, seed=999)
    model = RobustModel(config)

    logger.info(f"Model parameters: {model.count_parameters():,}")

    # Train
    server = BestPracticeServer(model, clients, test_data, config)
    history = server.train()

    # Summary
    print("\n" + "=" * 60)
    print("Training Complete")
    print(f"Final Accuracy: {history[-1]['accuracy']:.4f}")
    if server.failed_rounds:
        print(f"Failed Rounds: {server.failed_rounds}")
    print("=" * 60)


if __name__ == "__main__":
    main()
```

---

## Best Practices Checklist

### Before Training

- [ ] Validate configuration
- [ ] Initialize weights properly
- [ ] Create data loaders with shuffle
- [ ] Set random seeds for reproducibility

### During Training

- [ ] Deep copy models before local training
- [ ] Apply gradient clipping
- [ ] Use weighted aggregation
- [ ] Handle client failures gracefully

### During Evaluation

- [ ] Set model to eval mode
- [ ] Disable gradient computation
- [ ] Use held-out test data

---

## Exercises

1. **Exercise 1**: Add automatic hyperparameter validation
2. **Exercise 2**: Implement health checks for clients
3. **Exercise 3**: Add model checkpointing
4. **Exercise 4**: Create a stress test suite

---

## References

1. Kairouz, P., et al. (2021). Advances and open problems in FL. *FnTML*.
2. Clean Code by Robert C. Martin
3. Google Python Style Guide

---

*Copyright 2026 Olaf Yunus Laitinen Imanov and Contributors. Released under EUPL 1.2.*
