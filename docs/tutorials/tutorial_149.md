# Tutorial 149: FL Case Study Smart Cities

---

## Metadata

| Property | Value |
|----------|-------|
| **Tutorial ID** | 149 |
| **Title** | Federated Learning Case Study: Smart Cities |
| **Category** | Case Studies |
| **Difficulty** | Advanced |
| **Duration** | 120 minutes |
| **Prerequisites** | Tutorial 001-148 |
| **Author** | Unbitrium Contributors |
| **Last Updated** | January 2026 |

---

## Learning Objectives

By completing this tutorial, you will be able to:

1. **Understand** smart city FL applications and challenges
2. **Implement** traffic prediction using federated learning
3. **Design** privacy-preserving urban data analytics
4. **Analyze** multi-modal sensor data processing
5. **Deploy** FL systems for city-wide infrastructure

---

## Prerequisites

Before starting this tutorial, ensure you have:

- Completed tutorials 001-148
- Understanding of FL fundamentals
- Knowledge of time series prediction
- Familiarity with IoT systems

---

## Background and Theory

### Smart Cities and Data Privacy

Smart cities generate massive amounts of data from:
- Traffic sensors and cameras
- Environmental monitors
- Energy grid meters
- Public transportation systems
- Emergency services

Traditional centralized analytics raises privacy concerns. FL enables:
- Local data processing
- Privacy preservation
- Reduced bandwidth requirements
- Real-time analytics

### Smart City FL Architecture

```
Smart City FL Architecture:
├── Edge Tier
│   ├── Traffic sensors
│   ├── Air quality monitors
│   ├── Smart meters
│   └── Surveillance systems
├── Fog/Edge Servers
│   ├── District aggregators
│   ├── Local model training
│   └── Real-time inference
└── Cloud Tier
    ├── Global aggregation
    ├── Model management
    └── City-wide analytics
```

### Application Domains

| Domain | Data Type | Privacy Level |
|--------|-----------|---------------|
| Traffic | Flow, speed data | Medium |
| Energy | Consumption patterns | High |
| Environment | Air quality, noise | Low |
| Public Safety | Incident reports | High |
| Healthcare | Emergency calls | Very High |

---

## Implementation Code

```python
#!/usr/bin/env python3
"""
Tutorial 149: Federated Learning Case Study - Smart Cities

This module implements a federated learning system for smart city
applications, focusing on traffic prediction across city districts.

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
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SmartCityConfig:
    """Configuration for smart city FL simulation."""

    # FL parameters
    num_rounds: int = 50
    num_districts: int = 10
    districts_per_round: int = 8

    # Model parameters
    sensor_dim: int = 24  # 24-hour time series
    hidden_dim: int = 64
    prediction_horizon: int = 6  # Predict 6 hours ahead

    # Training parameters
    learning_rate: float = 0.001
    batch_size: int = 32
    local_epochs: int = 5

    # Data parameters
    samples_per_district: int = 500
    noise_level: float = 0.1

    seed: int = 42


class TrafficDataset(Dataset):
    """Traffic flow dataset for a city district.

    Simulates realistic traffic patterns with:
    - Daily cycles (rush hours, night lulls)
    - Weekly patterns (weekday vs weekend)
    - District-specific characteristics

    Args:
        district_id: Unique identifier for the district.
        n: Number of samples.
        dim: Time series dimension (hours).
        seed: Random seed.
        district_type: Type of district (residential, commercial, industrial).
    """

    def __init__(
        self,
        district_id: int,
        n: int = 500,
        dim: int = 24,
        seed: int = 0,
        district_type: str = "mixed"
    ):
        np.random.seed(seed + district_id)

        self.district_id = district_id
        self.district_type = district_type
        self.n = n
        self.dim = dim

        # Generate traffic patterns
        self.x, self.y = self._generate_traffic_data()

        logger.debug(
            f"Created traffic dataset for district {district_id} "
            f"({district_type}) with {n} samples"
        )

    def _generate_traffic_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate synthetic traffic data with realistic patterns."""

        # Base pattern templates
        if self.district_type == "commercial":
            # Higher midday traffic
            base_pattern = np.array([
                0.1, 0.05, 0.05, 0.05, 0.1, 0.3,  # 0-5
                0.6, 0.9, 1.0, 0.95, 0.9, 0.95,   # 6-11
                1.0, 0.95, 0.9, 0.85, 0.9, 1.0,   # 12-17
                0.8, 0.5, 0.3, 0.2, 0.15, 0.1     # 18-23
            ])
        elif self.district_type == "residential":
            # Morning and evening peaks
            base_pattern = np.array([
                0.05, 0.03, 0.03, 0.05, 0.1, 0.3,  # 0-5
                0.7, 1.0, 0.9, 0.4, 0.3, 0.35,     # 6-11
                0.4, 0.35, 0.4, 0.5, 0.7, 0.9,     # 12-17
                1.0, 0.8, 0.5, 0.3, 0.15, 0.08     # 18-23
            ])
        elif self.district_type == "industrial":
            # Shift-based patterns
            base_pattern = np.array([
                0.1, 0.08, 0.08, 0.1, 0.2, 0.5,   # 0-5
                0.9, 1.0, 0.95, 0.9, 0.9, 0.9,    # 6-11
                0.85, 0.9, 0.95, 0.9, 0.8, 0.6,   # 12-17
                0.4, 0.25, 0.15, 0.12, 0.1, 0.1   # 18-23
            ])
        else:  # mixed
            base_pattern = np.array([
                0.08, 0.05, 0.05, 0.08, 0.15, 0.4,  # 0-5
                0.8, 0.95, 0.9, 0.7, 0.65, 0.7,     # 6-11
                0.75, 0.7, 0.7, 0.75, 0.85, 0.95,   # 12-17
                0.9, 0.6, 0.4, 0.25, 0.15, 0.1      # 18-23
            ])

        # Generate samples with variations
        x_data = []
        y_data = []

        for _ in range(self.n):
            # Add random variations
            scale = np.random.uniform(0.8, 1.2)
            shift = np.random.uniform(-0.1, 0.1)
            noise = np.random.randn(self.dim) * 0.1

            pattern = base_pattern * scale + shift + noise
            pattern = np.clip(pattern, 0, 1.5)

            # Target is sum of next 6 hours (normalized)
            target = pattern[6:12].sum() / 6

            x_data.append(pattern)
            y_data.append(target)

        x = torch.tensor(np.array(x_data), dtype=torch.float32)
        y = torch.tensor(np.array(y_data), dtype=torch.float32).unsqueeze(1)

        return x, y

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx]


class EnvironmentalDataset(Dataset):
    """Environmental sensor dataset (air quality, noise, etc.)."""

    def __init__(
        self,
        district_id: int,
        n: int = 500,
        dim: int = 24,
        seed: int = 0
    ):
        np.random.seed(seed + district_id * 100)

        # Environmental readings: PM2.5, PM10, NO2, O3, noise
        self.num_sensors = 5
        self.x = torch.randn(n, dim, self.num_sensors)

        # Add patterns (pollution peaks correlate with traffic)
        for i in range(n):
            traffic_pattern = torch.sin(
                torch.linspace(0, 4 * np.pi, dim)
            )
            for s in range(self.num_sensors):
                self.x[i, :, s] += traffic_pattern * (s + 1) * 0.1

        # Target: air quality index prediction
        self.y = self.x.mean(dim=[1, 2]).unsqueeze(1)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx].flatten(), self.y[idx]


class TrafficPredictionModel(nn.Module):
    """LSTM-based traffic prediction model.

    Uses sequential processing given the temporal nature of traffic data.
    """

    def __init__(self, config: SmartCityConfig):
        super().__init__()

        self.config = config

        # Temporal feature extraction
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=config.hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )

        # Prediction head
        self.predictor = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(config.hidden_dim // 2, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, seq_len).

        Returns:
            Predictions of shape (batch, 1).
        """
        # Reshape for LSTM: (batch, seq, features)
        x = x.unsqueeze(-1)

        # LSTM encoding
        lstm_out, (h_n, c_n) = self.lstm(x)

        # Use final hidden state
        features = h_n[-1]

        # Predict
        return self.predictor(features)


class SimpleFeedforwardModel(nn.Module):
    """Simple feedforward model for baseline comparison."""

    def __init__(self, config: SmartCityConfig):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(config.sensor_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(config.hidden_dim // 2, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DistrictClient:
    """FL client representing a city district.

    Manages local traffic data and model training for one district.
    """

    def __init__(
        self,
        district_id: int,
        dataset: TrafficDataset,
        config: SmartCityConfig
    ):
        self.district_id = district_id
        self.dataset = dataset
        self.config = config

        # Track local metrics
        self.training_history: List[Dict] = []

    @property
    def district_type(self) -> str:
        return self.dataset.district_type

    def train(self, model: nn.Module) -> Dict[str, Any]:
        """Train on local district data.

        Args:
            model: Global model to train locally.

        Returns:
            Update dictionary with state_dict and metrics.
        """
        # Create local copy
        local_model = copy.deepcopy(model)
        optimizer = torch.optim.Adam(
            local_model.parameters(),
            lr=self.config.learning_rate
        )

        loader = DataLoader(
            self.dataset,
            batch_size=self.config.batch_size,
            shuffle=True
        )

        local_model.train()
        total_loss = 0.0
        num_batches = 0

        for epoch in range(self.config.local_epochs):
            epoch_loss = 0.0
            for x, y in loader:
                optimizer.zero_grad()

                pred = local_model(x)
                loss = F.mse_loss(pred, y)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(local_model.parameters(), 1.0)
                optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

            total_loss += epoch_loss

        avg_loss = total_loss / num_batches

        self.training_history.append({
            "loss": avg_loss,
            "samples": len(self.dataset)
        })

        return {
            "state_dict": {
                k: v.cpu() for k, v in local_model.state_dict().items()
            },
            "num_samples": len(self.dataset),
            "avg_loss": avg_loss,
            "district_id": self.district_id,
            "district_type": self.district_type
        }


class CityAggregator:
    """Aggregator for city-wide model updates.

    Supports multiple aggregation strategies for urban data.
    """

    def __init__(self, strategy: str = "weighted"):
        self.strategy = strategy

    def aggregate(
        self,
        updates: List[Dict[str, Any]]
    ) -> Dict[str, torch.Tensor]:
        """Aggregate district updates.

        Args:
            updates: List of district updates.

        Returns:
            Aggregated state dictionary.
        """
        if not updates:
            raise ValueError("No updates to aggregate")

        if self.strategy == "weighted":
            return self._weighted_average(updates)
        elif self.strategy == "uniform":
            return self._uniform_average(updates)
        else:
            return self._weighted_average(updates)

    def _weighted_average(
        self,
        updates: List[Dict[str, Any]]
    ) -> Dict[str, torch.Tensor]:
        """Weight by number of samples."""
        total_samples = sum(u["num_samples"] for u in updates)

        new_state = {}
        for key in updates[0]["state_dict"]:
            new_state[key] = sum(
                (u["num_samples"] / total_samples) * u["state_dict"][key].float()
                for u in updates
            )

        return new_state

    def _uniform_average(
        self,
        updates: List[Dict[str, Any]]
    ) -> Dict[str, torch.Tensor]:
        """Equal weight for all districts."""
        n = len(updates)

        new_state = {}
        for key in updates[0]["state_dict"]:
            new_state[key] = sum(
                u["state_dict"][key].float() for u in updates
            ) / n

        return new_state


class SmartCityServer:
    """Central server for city-wide FL coordination.

    Manages global model and coordinates district training.
    """

    def __init__(
        self,
        model: nn.Module,
        districts: List[DistrictClient],
        test_data: TrafficDataset,
        config: SmartCityConfig
    ):
        self.model = model
        self.districts = districts
        self.test_data = test_data
        self.config = config

        self.aggregator = CityAggregator(strategy="weighted")
        self.history: List[Dict] = []

    def select_districts(self) -> List[DistrictClient]:
        """Select districts for current round."""
        n = min(self.config.districts_per_round, len(self.districts))
        indices = np.random.choice(len(self.districts), n, replace=False)
        return [self.districts[i] for i in indices]

    def evaluate(self) -> Dict[str, float]:
        """Evaluate global model on test data."""
        self.model.eval()
        loader = DataLoader(self.test_data, batch_size=64)

        total_mse = 0.0
        total_mae = 0.0
        count = 0

        with torch.no_grad():
            for x, y in loader:
                pred = self.model(x)

                mse = F.mse_loss(pred, y, reduction='sum').item()
                mae = F.l1_loss(pred, y, reduction='sum').item()

                total_mse += mse
                total_mae += mae
                count += len(y)

        return {
            "mse": total_mse / count,
            "rmse": np.sqrt(total_mse / count),
            "mae": total_mae / count
        }

    def train_round(self, round_num: int) -> Dict[str, Any]:
        """Execute one FL round."""
        # Select participating districts
        selected = self.select_districts()

        # Collect updates
        updates = []
        for district in selected:
            update = district.train(self.model)
            updates.append(update)

        # Aggregate
        new_state = self.aggregator.aggregate(updates)
        self.model.load_state_dict(new_state)

        # Evaluate
        metrics = self.evaluate()

        # Record
        record = {
            "round": round_num,
            **metrics,
            "num_districts": len(selected),
            "avg_train_loss": np.mean([u["avg_loss"] for u in updates])
        }
        self.history.append(record)

        return record

    def train(self) -> List[Dict]:
        """Run full FL training."""
        logger.info(f"Starting smart city FL with {len(self.districts)} districts")

        for round_num in range(self.config.num_rounds):
            record = self.train_round(round_num)

            if (round_num + 1) % 10 == 0:
                logger.info(
                    f"Round {round_num + 1}: "
                    f"RMSE={record['rmse']:.4f}, "
                    f"MAE={record['mae']:.4f}"
                )

        return self.history


def create_city_districts(
    config: SmartCityConfig
) -> List[DistrictClient]:
    """Create diverse city districts."""
    district_types = [
        "commercial", "residential", "industrial",
        "commercial", "residential", "mixed",
        "residential", "mixed", "commercial", "mixed"
    ]

    districts = []
    for i in range(config.num_districts):
        dtype = district_types[i % len(district_types)]
        dataset = TrafficDataset(
            district_id=i,
            n=config.samples_per_district,
            dim=config.sensor_dim,
            seed=config.seed,
            district_type=dtype
        )
        client = DistrictClient(i, dataset, config)
        districts.append(client)

    return districts


def main():
    """Main entry point."""
    print("=" * 60)
    print("Tutorial 149: FL Case Study - Smart Cities")
    print("=" * 60)

    config = SmartCityConfig()
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    # Create city districts
    districts = create_city_districts(config)

    # Create test data
    test_data = TrafficDataset(
        district_id=999,
        n=300,
        dim=config.sensor_dim,
        seed=999,
        district_type="mixed"
    )

    # Create model
    model = SimpleFeedforwardModel(config)
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create server and train
    server = SmartCityServer(model, districts, test_data, config)
    history = server.train()

    # Summary
    print("\n" + "=" * 60)
    print("Training Complete")
    print(f"Final RMSE: {history[-1]['rmse']:.4f}")
    print(f"Final MAE: {history[-1]['mae']:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
```

---

## Key Insights

### Smart City FL Challenges

1. **Heterogeneous Data**: Different district types have different patterns
2. **Real-time Requirements**: Traffic prediction needs low latency
3. **Scalability**: Cities may have hundreds of sensors
4. **Privacy**: Citizen movement patterns are sensitive

### Best Practices

- Use hierarchical aggregation for large-scale deployments
- Consider temporal patterns in model design
- Implement edge caching for frequent predictions
- Monitor model drift as city patterns evolve

---

## Exercises

1. **Exercise 1**: Add environmental data fusion
2. **Exercise 2**: Implement hierarchical city aggregation
3. **Exercise 3**: Add real-time anomaly detection
4. **Exercise 4**: Design a cross-modal sensor fusion model

---

## References

1. Liu, Y., et al. (2020). Privacy-preserving traffic flow prediction. *IEEE TKDE*.
2. Jiang, J.C., et al. (2020). FL for smart city applications. *IEEE IoT Journal*.
3. Zheng, Y., et al. (2014). Urban computing: Concepts, methodologies, and applications. *ACM TIST*.

---

*Copyright 2026 Olaf Yunus Laitinen Imanov and Contributors. Released under EUPL 1.2.*
