# Tutorial 150: FL Case Study Manufacturing

---

## Metadata

| Property | Value |
|----------|-------|
| **Tutorial ID** | 150 |
| **Title** | Federated Learning Case Study: Manufacturing |
| **Category** | Case Studies |
| **Difficulty** | Advanced |
| **Duration** | 120 minutes |
| **Prerequisites** | Tutorial 001-149 |
| **Author** | Unbitrium Contributors |
| **Last Updated** | January 2026 |

---

## Learning Objectives

By completing this tutorial, you will be able to:

1. **Understand** manufacturing FL applications and requirements
2. **Implement** predictive maintenance using FL
3. **Design** cross-factory model sharing systems
4. **Analyze** industrial sensor data patterns
5. **Deploy** FL for quality control applications

---

## Prerequisites

Before starting this tutorial, ensure you have:

- Completed tutorials 001-149
- Understanding of FL fundamentals
- Knowledge of time series analysis
- Familiarity with industrial IoT systems

---

## Background and Theory

### Manufacturing and Industry 4.0

Modern manufacturing generates vast amounts of sensor data:
- Machine vibration and acoustic signals
- Temperature and pressure readings
- Production quality metrics
- Energy consumption data

FL enables manufacturers to:
- Share learnings without revealing proprietary processes
- Build robust models across factory variations
- Maintain data sovereignty and security
- Comply with data localization requirements

### Manufacturing FL Architecture

```
Manufacturing FL Architecture:
├── Machine Level
│   ├── Sensors (vibration, temp, etc.)
│   ├── PLCs and edge devices
│   └── Local preprocessing
├── Factory Level
│   ├── Factory aggregator
│   ├── Quality control systems
│   └── Local model training
└── Enterprise Level
    ├── Global model coordination
    ├── Cross-factory learning
    └── Fleet management
```

### Key Applications

| Application | Data Type | Value Proposition |
|-------------|-----------|-------------------|
| Predictive Maintenance | Sensor signals | Reduce downtime 30-50% |
| Quality Control | Inspection data | Reduce defects |
| Process Optimization | Production params | Improve efficiency |
| Energy Management | Power consumption | Reduce costs |

---

## Implementation Code

```python
#!/usr/bin/env python3
"""
Tutorial 150: Federated Learning Case Study - Manufacturing

This module implements a federated learning system for manufacturing
applications, focusing on predictive maintenance across factories.

Copyright 2026 Olaf Yunus Laitinen Imanov and Contributors
Released under EUPL 1.2
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import copy
import logging
from enum import Enum

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MachineState(Enum):
    """Machine health states."""
    HEALTHY = 0
    DEGRADED = 1
    FAILURE_IMMINENT = 2


@dataclass
class ManufacturingConfig:
    """Configuration for manufacturing FL."""

    # FL parameters
    num_rounds: int = 50
    num_factories: int = 8
    factories_per_round: int = 6

    # Model parameters
    sensor_dim: int = 20  # Number of sensor readings
    hidden_dim: int = 64
    num_classes: int = 3  # health states

    # Training parameters
    learning_rate: float = 0.001
    batch_size: int = 32
    local_epochs: int = 5

    # Data parameters
    samples_per_factory: int = 500
    failure_rate: float = 0.15  # 15% failure samples

    seed: int = 42


class MachineDataset(Dataset):
    """Machine sensor dataset for predictive maintenance.

    Simulates sensor data from industrial machines with:
    - Vibration signatures
    - Temperature readings
    - Pressure measurements
    - Current/voltage data

    Each factory has slightly different machine characteristics.
    """

    def __init__(
        self,
        factory_id: int,
        n: int = 500,
        sensor_dim: int = 20,
        failure_rate: float = 0.15,
        seed: int = 0
    ):
        np.random.seed(seed + factory_id)

        self.factory_id = factory_id
        self.n = n
        self.sensor_dim = sensor_dim

        # Generate sensor data
        self.x, self.y = self._generate_data(failure_rate)

        # Factory-specific scaling (different machines/sensors)
        self.factory_scale = 0.9 + 0.2 * np.random.random()
        self.x = self.x * self.factory_scale

        logger.debug(f"Created factory {factory_id} dataset with {n} samples")

    def _generate_data(
        self,
        failure_rate: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate synthetic machine sensor data."""

        x_data = []
        y_data = []

        for i in range(self.n):
            # Determine machine state
            r = np.random.random()
            if r < failure_rate:
                state = MachineState.FAILURE_IMMINENT.value
            elif r < failure_rate + 0.2:
                state = MachineState.DEGRADED.value
            else:
                state = MachineState.HEALTHY.value

            # Generate sensor readings based on state
            sensors = self._generate_sensor_reading(state)

            x_data.append(sensors)
            y_data.append(state)

        x = torch.tensor(np.array(x_data), dtype=torch.float32)
        y = torch.tensor(y_data, dtype=torch.long)

        return x, y

    def _generate_sensor_reading(self, state: int) -> np.ndarray:
        """Generate sensor reading for a given machine state."""

        # Base noise level
        sensors = np.random.randn(self.sensor_dim) * 0.1

        if state == MachineState.HEALTHY.value:
            # Normal operation: low variance, centered
            sensors += np.random.randn(self.sensor_dim) * 0.05

        elif state == MachineState.DEGRADED.value:
            # Degradation: increased variance, slight drift
            sensors += np.random.randn(self.sensor_dim) * 0.2
            sensors[0:5] += 0.5  # Vibration increase
            sensors[5:10] += 0.3  # Temperature rise

        else:  # FAILURE_IMMINENT
            # Near failure: high variance, strong drift
            sensors += np.random.randn(self.sensor_dim) * 0.4
            sensors[0:5] += 1.5  # Strong vibration
            sensors[5:10] += 1.0  # High temperature
            sensors[10:15] += 0.8  # Pressure anomaly

        return sensors

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx]


class PredictiveMaintenanceModel(nn.Module):
    """Neural network for machine health prediction.

    Uses a multi-layer architecture with attention to important sensors.
    """

    def __init__(self, config: ManufacturingConfig):
        super().__init__()

        self.config = config

        # Sensor attention layer
        self.attention = nn.Sequential(
            nn.Linear(config.sensor_dim, config.sensor_dim),
            nn.Tanh(),
            nn.Linear(config.sensor_dim, config.sensor_dim),
            nn.Softmax(dim=-1)
        )

        # Feature extraction
        self.encoder = nn.Sequential(
            nn.Linear(config.sensor_dim, config.hidden_dim),
            nn.BatchNorm1d(config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.BatchNorm1d(config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        # Classifier
        self.classifier = nn.Linear(config.hidden_dim, config.num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with attention."""
        # Compute attention weights
        attn = self.attention(x)

        # Apply attention
        x_attended = x * attn

        # Encode
        features = self.encoder(x_attended)

        # Classify
        return self.classifier(features)

    def get_attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        """Get attention weights for interpretability."""
        with torch.no_grad():
            return self.attention(x)


class QualityMetrics:
    """Metrics for manufacturing quality assessment."""

    @staticmethod
    def compute_metrics(
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> Dict[str, float]:
        """Compute comprehensive classification metrics."""

        pred_labels = predictions.argmax(dim=1)

        # Overall accuracy
        accuracy = (pred_labels == targets).float().mean().item()

        # Per-class recall (important for failure detection)
        recalls = {}
        for state in MachineState:
            mask = targets == state.value
            if mask.sum() > 0:
                correct = ((pred_labels == state.value) & mask).sum()
                recalls[f"recall_{state.name.lower()}"] = (
                    correct / mask.sum()
                ).item()

        # Failure detection rate (most critical metric)
        failure_mask = targets == MachineState.FAILURE_IMMINENT.value
        if failure_mask.sum() > 0:
            failure_detected = (
                (pred_labels == MachineState.FAILURE_IMMINENT.value) & failure_mask
            ).sum()
            failure_detection_rate = (failure_detected / failure_mask.sum()).item()
        else:
            failure_detection_rate = 0.0

        return {
            "accuracy": accuracy,
            "failure_detection_rate": failure_detection_rate,
            **recalls
        }


class FactoryClient:
    """FL client representing a factory."""

    def __init__(
        self,
        factory_id: int,
        dataset: MachineDataset,
        config: ManufacturingConfig
    ):
        self.factory_id = factory_id
        self.dataset = dataset
        self.config = config

        self.training_history: List[Dict] = []

    def train(self, model: nn.Module) -> Dict[str, Any]:
        """Train on local factory data."""

        local_model = copy.deepcopy(model)
        optimizer = torch.optim.Adam(
            local_model.parameters(),
            lr=self.config.learning_rate
        )

        # Class weights for imbalanced data
        class_weights = torch.tensor([1.0, 2.0, 5.0])  # Prioritize failures

        loader = DataLoader(
            self.dataset,
            batch_size=self.config.batch_size,
            shuffle=True
        )

        local_model.train()
        total_loss = 0.0
        num_batches = 0

        for epoch in range(self.config.local_epochs):
            for x, y in loader:
                optimizer.zero_grad()

                output = local_model(x)
                loss = F.cross_entropy(output, y, weight=class_weights)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(local_model.parameters(), 1.0)
                optimizer.step()

                total_loss += loss.item()
                num_batches += 1

        avg_loss = total_loss / num_batches

        return {
            "state_dict": {
                k: v.cpu() for k, v in local_model.state_dict().items()
            },
            "num_samples": len(self.dataset),
            "avg_loss": avg_loss,
            "factory_id": self.factory_id
        }


class EnterpriseServer:
    """Enterprise-level FL server for multi-factory coordination."""

    def __init__(
        self,
        model: nn.Module,
        factories: List[FactoryClient],
        test_data: MachineDataset,
        config: ManufacturingConfig
    ):
        self.model = model
        self.factories = factories
        self.test_data = test_data
        self.config = config

        self.history: List[Dict] = []
        self.best_detection_rate = 0.0

    def aggregate(self, updates: List[Dict[str, Any]]) -> None:
        """Aggregate factory updates."""
        if not updates:
            return

        total_samples = sum(u["num_samples"] for u in updates)
        new_state = {}

        for key in updates[0]["state_dict"]:
            new_state[key] = sum(
                (u["num_samples"] / total_samples) * u["state_dict"][key].float()
                for u in updates
            )

        self.model.load_state_dict(new_state)

    def evaluate(self) -> Dict[str, float]:
        """Evaluate on test data."""
        self.model.eval()
        loader = DataLoader(self.test_data, batch_size=64)

        all_preds = []
        all_targets = []

        with torch.no_grad():
            for x, y in loader:
                output = self.model(x)
                all_preds.append(output)
                all_targets.append(y)

        predictions = torch.cat(all_preds)
        targets = torch.cat(all_targets)

        return QualityMetrics.compute_metrics(predictions, targets)

    def train(self) -> List[Dict]:
        """Run federated training."""
        logger.info(f"Starting manufacturing FL with {len(self.factories)} factories")

        for round_num in range(self.config.num_rounds):
            # Select factories
            n = min(self.config.factories_per_round, len(self.factories))
            indices = np.random.choice(len(self.factories), n, replace=False)
            selected = [self.factories[i] for i in indices]

            # Collect updates
            updates = [f.train(self.model) for f in selected]

            # Aggregate
            self.aggregate(updates)

            # Evaluate
            metrics = self.evaluate()

            # Track best
            if metrics["failure_detection_rate"] > self.best_detection_rate:
                self.best_detection_rate = metrics["failure_detection_rate"]

            record = {
                "round": round_num,
                **metrics,
                "num_factories": len(selected),
                "avg_train_loss": np.mean([u["avg_loss"] for u in updates])
            }
            self.history.append(record)

            if (round_num + 1) % 10 == 0:
                logger.info(
                    f"Round {round_num + 1}: "
                    f"acc={metrics['accuracy']:.4f}, "
                    f"failure_det={metrics['failure_detection_rate']:.4f}"
                )

        return self.history


def main():
    """Main entry point."""
    print("=" * 60)
    print("Tutorial 150: FL Case Study - Manufacturing")
    print("=" * 60)

    config = ManufacturingConfig()
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    # Create factory datasets
    factories = []
    for i in range(config.num_factories):
        dataset = MachineDataset(
            factory_id=i,
            n=config.samples_per_factory,
            sensor_dim=config.sensor_dim,
            failure_rate=config.failure_rate,
            seed=config.seed
        )
        client = FactoryClient(i, dataset, config)
        factories.append(client)

    # Test data
    test_data = MachineDataset(
        factory_id=999,
        n=300,
        seed=999
    )

    # Model
    model = PredictiveMaintenanceModel(config)
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Train
    server = EnterpriseServer(model, factories, test_data, config)
    history = server.train()

    # Summary
    print("\n" + "=" * 60)
    print("Training Complete")
    print(f"Final Accuracy: {history[-1]['accuracy']:.4f}")
    print(f"Failure Detection Rate: {history[-1]['failure_detection_rate']:.4f}")
    print(f"Best Failure Detection: {server.best_detection_rate:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
```

---

## Key Insights

### Manufacturing FL Challenges

1. **Class Imbalance**: Failures are rare but critical to detect
2. **Domain Shift**: Different factories have different machines
3. **Real-time Requirements**: Predictions must be fast
4. **Explainability**: Need to understand why a failure is predicted

### Best Practices

- Weight loss function to prioritize failure detection
- Use attention mechanisms for sensor importance
- Implement continuous monitoring pipelines
- Validate across different factory types

---

## Exercises

1. **Exercise 1**: Add remaining useful life (RUL) prediction
2. **Exercise 2**: Implement factory-specific personalization
3. **Exercise 3**: Add anomaly detection for novel failures
4. **Exercise 4**: Design multi-modal sensor fusion

---

## References

1. Zhang, Q., et al. (2021). FL for predictive maintenance. *IEEE TASE*.
2. Hao, M., et al. (2019). Privacy-preserving FL in fog computing. *IEEE TII*.
3. Susto, G.A., et al. (2015). Machine learning for predictive maintenance. *IEEE TII*.

---

*Copyright 2026 Olaf Yunus Laitinen Imanov and Contributors. Released under EUPL 1.2.*
