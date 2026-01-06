# Tutorial 137: FL 5G Networks

---

## Metadata

| Property | Value |
|----------|-------|
| **Tutorial ID** | 137 |
| **Title** | Federated Learning for 5G Networks |
| **Category** | Infrastructure |
| **Difficulty** | Advanced |
| **Duration** | 120 minutes |
| **Prerequisites** | Tutorial 001-136 |
| **Author** | Unbitrium Contributors |
| **Last Updated** | January 2026 |

---

## Learning Objectives

By completing this tutorial, you will be able to:

1. **Understand** FL applications in 5G/6G networks
2. **Implement** network traffic prediction with FL
3. **Design** MEC-based FL systems
4. **Analyze** ultra-low latency requirements
5. **Deploy** FL for network optimization

---

## Prerequisites

Before starting this tutorial, ensure you have:

- Completed tutorials 001-136
- Understanding of FL fundamentals
- Knowledge of network systems
- Familiarity with mobile edge computing

---

## Background and Theory

### 5G and Federated Learning

5G networks generate massive amounts of data at the edge:
- User mobility patterns
- Channel quality measurements
- Traffic load statistics
- Application usage data

FL enables:
- Privacy-preserving network optimization
- Edge-based intelligence
- Reduced backhaul traffic
- Real-time network adaptation

### 5G FL Architecture

```
5G FL Architecture:
├── User Equipment (UE)
│   ├── Local data collection
│   ├── On-device inference
│   └── Privacy-preserving update
├── Mobile Edge Computing (MEC)
│   ├── Edge aggregation
│   ├── Low-latency inference
│   └── Resource management
├── Radio Access Network (RAN)
│   ├── BBU processing
│   ├── Traffic prediction
│   └── Resource allocation
└── Core Network
    ├── Global aggregation
    ├── Network slice management
    └── Policy optimization
```

### Key Applications

| Application | Latency Requirement | FL Benefit |
|-------------|-------------------|------------|
| Traffic Prediction | <100ms | Local patterns |
| Handover Optimization | <10ms | Mobility learning |
| Resource Allocation | <50ms | Demand prediction |
| Anomaly Detection | <1s | Distributed detection |

---

## Implementation Code

```python
#!/usr/bin/env python3
"""
Tutorial 137: Federated Learning for 5G Networks

This module implements FL for 5G network optimization,
focusing on traffic prediction at the edge.

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


class NetworkSliceType(Enum):
    """5G network slice types."""
    EMBB = "enhanced_mobile_broadband"
    URLLC = "ultra_reliable_low_latency"
    MMTC = "massive_machine_type"


@dataclass
class Network5GConfig:
    """Configuration for 5G FL."""

    # FL parameters
    num_rounds: int = 50
    num_base_stations: int = 20
    bs_per_round: int = 15

    # Model parameters
    input_features: int = 24  # 24-hour history
    hidden_dim: int = 64
    prediction_horizon: int = 6

    # Training parameters
    learning_rate: float = 0.001
    batch_size: int = 32
    local_epochs: int = 3

    # 5G parameters
    bandwidth_mhz: float = 100.0
    subcarrier_spacing_khz: float = 30.0

    # Data parameters
    samples_per_bs: int = 500

    seed: int = 42


class NetworkTrafficDataset(Dataset):
    """Network traffic dataset for a base station.

    Features include:
    - Historical traffic load
    - User count
    - Time-of-day patterns
    - Day-of-week patterns
    """

    def __init__(
        self,
        bs_id: int,
        n: int = 500,
        input_features: int = 24,
        seed: int = 0,
        slice_type: NetworkSliceType = NetworkSliceType.EMBB
    ):
        np.random.seed(seed + bs_id)

        self.bs_id = bs_id
        self.slice_type = slice_type

        # Generate traffic patterns
        self.x, self.y = self._generate_traffic_data(n, input_features)

    def _generate_traffic_data(
        self,
        n: int,
        features: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate synthetic network traffic data."""

        x_data = []
        y_data = []

        # Base pattern depends on slice type
        if self.slice_type == NetworkSliceType.EMBB:
            base_pattern = self._embb_pattern(features)
        elif self.slice_type == NetworkSliceType.URLLC:
            base_pattern = self._urllc_pattern(features)
        else:
            base_pattern = self._mmtc_pattern(features)

        for _ in range(n):
            # Add variations
            scale = np.random.uniform(0.7, 1.3)
            noise = np.random.randn(features) * 0.1

            traffic = base_pattern * scale + noise
            traffic = np.clip(traffic, 0, 1)

            # Target: next 6-hour average
            target = traffic[features//2:features//2+6].mean()

            x_data.append(traffic)
            y_data.append(target)

        x = torch.tensor(np.array(x_data), dtype=torch.float32)
        y = torch.tensor(y_data, dtype=torch.float32).unsqueeze(1)

        return x, y

    def _embb_pattern(self, features: int) -> np.ndarray:
        """Enhanced Mobile Broadband pattern (video, data)."""
        hours = np.arange(features)
        # Peak during evening hours
        pattern = (
            0.3 + 0.5 * np.exp(-((hours - 20) ** 2) / 20) +
            0.3 * np.exp(-((hours - 12) ** 2) / 30)
        )
        return pattern / pattern.max()

    def _urllc_pattern(self, features: int) -> np.ndarray:
        """Ultra-Reliable Low Latency pattern (industrial)."""
        hours = np.arange(features)
        # Business hours pattern
        pattern = np.where(
            (hours >= 8) & (hours <= 18),
            0.8 + 0.1 * np.random.randn(features),
            0.2 + 0.05 * np.random.randn(features)
        )
        return np.clip(pattern, 0, 1)

    def _mmtc_pattern(self, features: int) -> np.ndarray:
        """Massive Machine Type pattern (IoT)."""
        # More uniform with periodic spikes
        pattern = 0.4 + 0.2 * np.sin(np.linspace(0, 4*np.pi, features))
        pattern += np.random.randn(features) * 0.1
        return np.clip(pattern, 0, 1)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx]


class TrafficPredictor(nn.Module):
    """Traffic prediction model for 5G networks."""

    def __init__(self, config: Network5GConfig):
        super().__init__()

        self.config = config

        # Temporal convolution
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)

        # LSTM for sequence
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=config.hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )

        # Prediction head
        self.fc = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(config.hidden_dim // 2, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        # Conv layers expect (batch, channels, seq)
        x = x.unsqueeze(1)

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        # LSTM expects (batch, seq, features)
        x = x.permute(0, 2, 1)

        lstm_out, (h_n, c_n) = self.lstm(x)

        # Use final hidden state
        output = self.fc(h_n[-1])

        return output


class BaseStationClient:
    """FL client representing a base station."""

    def __init__(
        self,
        bs_id: int,
        dataset: NetworkTrafficDataset,
        config: Network5GConfig
    ):
        self.bs_id = bs_id
        self.dataset = dataset
        self.config = config

        # Resource constraints
        self.compute_capacity = np.random.uniform(0.5, 1.0)

    def train(self, model: nn.Module) -> Dict[str, Any]:
        """Train on local network data."""

        # Adjust epochs based on compute capacity
        epochs = max(1, int(self.config.local_epochs * self.compute_capacity))

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

        for _ in range(epochs):
            for x, y in loader:
                optimizer.zero_grad()
                pred = local_model(x)
                loss = F.mse_loss(pred, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(local_model.parameters(), 1.0)
                optimizer.step()

                total_loss += loss.item()
                num_batches += 1

        return {
            "state_dict": {
                k: v.cpu() for k, v in local_model.state_dict().items()
            },
            "num_samples": len(self.dataset),
            "avg_loss": total_loss / num_batches,
            "bs_id": self.bs_id,
            "slice_type": self.dataset.slice_type.value
        }


class MECServer:
    """Mobile Edge Computing server for FL aggregation."""

    def __init__(
        self,
        model: nn.Module,
        base_stations: List[BaseStationClient],
        test_data: NetworkTrafficDataset,
        config: Network5GConfig
    ):
        self.model = model
        self.base_stations = base_stations
        self.test_data = test_data
        self.config = config
        self.history: List[Dict] = []

    def aggregate(self, updates: List[Dict]) -> None:
        """Aggregate base station updates."""
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
        """Evaluate prediction accuracy."""
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

    def train(self) -> List[Dict]:
        """Run FL training."""
        logger.info(f"Starting 5G FL with {len(self.base_stations)} base stations")

        for round_num in range(self.config.num_rounds):
            # Select base stations
            n = min(self.config.bs_per_round, len(self.base_stations))
            indices = np.random.choice(len(self.base_stations), n, replace=False)
            selected = [self.base_stations[i] for i in indices]

            # Collect updates
            updates = [bs.train(self.model) for bs in selected]

            # Aggregate
            self.aggregate(updates)

            # Evaluate
            metrics = self.evaluate()

            record = {
                "round": round_num,
                **metrics,
                "num_bs": len(selected),
                "avg_loss": np.mean([u["avg_loss"] for u in updates])
            }
            self.history.append(record)

            if (round_num + 1) % 10 == 0:
                logger.info(
                    f"Round {round_num + 1}: "
                    f"RMSE={metrics['rmse']:.4f}, "
                    f"MAE={metrics['mae']:.4f}"
                )

        return self.history


def main():
    """Main entry point."""
    print("=" * 60)
    print("Tutorial 137: FL for 5G Networks")
    print("=" * 60)

    config = Network5GConfig()
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    # Create base stations with different slice types
    slice_types = [NetworkSliceType.EMBB, NetworkSliceType.URLLC, NetworkSliceType.MMTC]
    base_stations = []

    for i in range(config.num_base_stations):
        slice_type = slice_types[i % 3]
        dataset = NetworkTrafficDataset(
            bs_id=i,
            n=config.samples_per_bs,
            input_features=config.input_features,
            seed=config.seed,
            slice_type=slice_type
        )
        client = BaseStationClient(i, dataset, config)
        base_stations.append(client)

    # Test data
    test_data = NetworkTrafficDataset(
        bs_id=999,
        n=500,
        seed=999
    )

    # Model
    model = TrafficPredictor(config)
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Train
    server = MECServer(model, base_stations, test_data, config)
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

### 5G FL Challenges

1. **Latency Requirements**: Ultra-low latency for URLLC
2. **Heterogeneous Traffic**: Different slice types
3. **Mobility**: Users move between cells
4. **Resource Constraints**: Edge compute limits

### Best Practices

- Adapt training to compute capacity
- Consider network slice types
- Use hierarchical aggregation
- Optimize for edge deployment

---

## Exercises

1. **Exercise 1**: Add handover prediction
2. **Exercise 2**: Implement resource allocation
3. **Exercise 3**: Add network slice personalization
4. **Exercise 4**: Design latency-aware aggregation

---

## References

1. Niknam, S., et al. (2020). FL for wireless communications. *IEEE COMST*.
2. Wen, W., et al. (2020). Joint scheduling and resource allocation for FL in 5G. *IEEE JSAC*.
3. Lim, W.Y.B., et al. (2020). FL in mobile edge networks. *IEEE COMST*.

---

*Copyright 2026 Olaf Yunus Laitinen Imanov and Contributors. Released under EUPL 1.2.*
