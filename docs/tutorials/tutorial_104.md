# Tutorial 104: FL Cross-Device Systems

---

## Metadata

| Property | Value |
|----------|-------|
| **Tutorial ID** | 104 |
| **Title** | FL Cross-Device Systems |
| **Category** | Systems |
| **Difficulty** | Advanced |
| **Duration** | 120 minutes |
| **Prerequisites** | Tutorial 001-103 |
| **Author** | Unbitrium Contributors |
| **Last Updated** | January 2026 |

---

## Learning Objectives

By completing this tutorial, you will be able to:

1. **Understand** cross-device FL
2. **Implement** mobile FL protocols
3. **Design** large-scale systems
4. **Handle** device heterogeneity
5. **Deploy** production cross-device FL

---

## Background and Theory

### Cross-Device Characteristics

```
Cross-Device FL:
├── Scale
│   ├── Millions of devices
│   ├── Partial participation
│   └── Anonymous clients
├── Constraints
│   ├── Limited bandwidth
│   ├── Battery constraints
│   ├── Intermittent connectivity
│   └── Local compute limits
├── Privacy
│   ├── On-device training only
│   ├── No data extraction
│   └── Secure aggregation
└── Coordination
    ├── Async training
    ├── Selection criteria
    └── Update scheduling
```

---

## Implementation Code

```python
#!/usr/bin/env python3
"""
Tutorial 104: FL Cross-Device Systems

This module implements cross-device federated
learning for mobile and edge devices.

Copyright 2026 Olaf Yunus Laitinen Imanov and Contributors
Released under EUPL 1.2
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from enum import Enum
import copy
import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DeviceState(Enum):
    IDLE = "idle"
    TRAINING = "training"
    UPLOADING = "uploading"
    UNAVAILABLE = "unavailable"


@dataclass
class DeviceProfile:
    """Device characteristics."""

    device_id: int
    battery_level: float = 1.0
    network_bandwidth: float = 1.0
    compute_capacity: float = 1.0
    available: bool = True

    def update(self):
        """Simulate device state changes."""
        self.battery_level = max(0, self.battery_level - np.random.uniform(0, 0.05))
        self.available = self.battery_level > 0.2 and np.random.random() > 0.1


@dataclass
class CrossDeviceConfig:
    """Cross-device configuration."""

    num_rounds: int = 50
    population_size: int = 1000  # Total devices
    cohort_size: int = 100       # Selected per round
    target_participants: int = 50 # Actually participate

    input_dim: int = 32
    hidden_dim: int = 32  # Small model for mobile
    num_classes: int = 10

    learning_rate: float = 0.01
    batch_size: int = 16
    local_epochs: int = 1

    seed: int = 42


class MobileModel(nn.Module):
    """Lightweight model for mobile devices."""

    def __init__(self, config: CrossDeviceConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.num_classes)
        )

    def forward(self, x): return self.net(x)


class DeviceDataset(Dataset):
    """On-device dataset."""

    def __init__(self, device_id: int, n: int = 50, dim: int = 32, classes: int = 10, seed: int = 0):
        np.random.seed(seed + device_id)
        self.x = torch.randn(n, dim, dtype=torch.float32)
        self.y = torch.randint(0, classes, (n,), dtype=torch.long)
        for i in range(n):
            self.x[i, self.y[i].item() % dim] += 2.0

    def __len__(self): return len(self.y)
    def __getitem__(self, idx): return self.x[idx], self.y[idx]


class MobileDevice:
    """Mobile device in cross-device FL."""

    def __init__(self, device_id: int, config: CrossDeviceConfig):
        self.device_id = device_id
        self.config = config

        self.profile = DeviceProfile(device_id)
        self.dataset = DeviceDataset(device_id, n=30 + np.random.randint(0, 50), seed=config.seed)
        self.state = DeviceState.IDLE

    def check_eligibility(self) -> bool:
        """Check if device can participate."""
        self.profile.update()

        return (
            self.profile.available and
            self.profile.battery_level > 0.3 and
            self.state == DeviceState.IDLE
        )

    def train(self, model: nn.Module) -> Optional[Dict]:
        """On-device training."""
        if not self.check_eligibility():
            return None

        self.state = DeviceState.TRAINING

        local = copy.deepcopy(model)
        optimizer = torch.optim.SGD(local.parameters(), lr=self.config.learning_rate)
        loader = DataLoader(self.dataset, batch_size=self.config.batch_size, shuffle=True)

        local.train()
        total_loss = 0.0
        num_batches = 0

        for _ in range(self.config.local_epochs):
            for x, y in loader:
                optimizer.zero_grad()
                loss = F.cross_entropy(local(x), y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                num_batches += 1

        self.state = DeviceState.UPLOADING

        # Simulate upload latency based on bandwidth
        time.sleep(0.001 / self.profile.network_bandwidth)

        self.state = DeviceState.IDLE
        self.profile.battery_level -= 0.02

        return {
            "state_dict": {k: v.cpu() for k, v in local.state_dict().items()},
            "num_samples": len(self.dataset),
            "device_id": self.device_id
        }


class DeviceSelector:
    """Select devices for training."""

    def __init__(self, config: CrossDeviceConfig):
        self.config = config

    def select(self, devices: List[MobileDevice]) -> List[int]:
        """Select cohort of devices."""
        # Filter eligible
        eligible = [d for d in devices if d.check_eligibility()]

        if len(eligible) < self.config.target_participants:
            return [d.device_id for d in eligible]

        # Score devices
        scores = []
        for d in eligible:
            score = (
                d.profile.battery_level * 0.3 +
                d.profile.compute_capacity * 0.4 +
                d.profile.network_bandwidth * 0.3
            )
            scores.append((d.device_id, score))

        # Select top devices
        scores.sort(key=lambda x: x[1], reverse=True)
        return [s[0] for s in scores[:self.config.cohort_size]]


class CrossDeviceServer:
    """Server for cross-device FL."""

    def __init__(self, model: nn.Module, devices: List[MobileDevice], test_data: DeviceDataset, config: CrossDeviceConfig):
        self.model = model
        self.devices = devices
        self.test_data = test_data
        self.config = config

        self.selector = DeviceSelector(config)
        self.device_map = {d.device_id: d for d in devices}
        self.history: List[Dict] = []

    def aggregate(self, updates: List[Dict]) -> None:
        total = sum(u["num_samples"] for u in updates)
        new_state = {}
        for key in updates[0]["state_dict"]:
            new_state[key] = sum(
                (u["num_samples"] / total) * u["state_dict"][key].float()
                for u in updates
            )
        self.model.load_state_dict(new_state)

    def evaluate(self) -> Dict[str, float]:
        self.model.eval()
        loader = DataLoader(self.test_data, batch_size=64)
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in loader:
                pred = self.model(x).argmax(dim=1)
                correct += (pred == y).sum().item()
                total += len(y)
        return {"accuracy": correct / total}

    def train(self) -> List[Dict]:
        logger.info(f"Starting cross-device FL ({len(self.devices)} devices)")

        for round_num in range(self.config.num_rounds):
            selected_ids = self.selector.select(self.devices)
            selected = [self.device_map[i] for i in selected_ids if i in self.device_map]

            updates = [d.train(self.model) for d in selected[:self.config.target_participants]]
            updates = [u for u in updates if u is not None]

            if updates:
                self.aggregate(updates)

            metrics = self.evaluate()

            record = {
                "round": round_num,
                "participants": len(updates),
                **metrics
            }
            self.history.append(record)

            if (round_num + 1) % 10 == 0:
                logger.info(f"Round {round_num + 1}: acc={metrics['accuracy']:.4f}, participants={len(updates)}")

        return self.history


def main():
    print("=" * 60)
    print("Tutorial 104: FL Cross-Device Systems")
    print("=" * 60)

    config = CrossDeviceConfig()
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    devices = [MobileDevice(i, config) for i in range(config.population_size)]
    test_data = DeviceDataset(device_id=9999, n=500, seed=999)
    model = MobileModel(config)

    server = CrossDeviceServer(model, devices, test_data, config)
    history = server.train()

    avg_participants = np.mean([h["participants"] for h in history])

    print("\n" + "=" * 60)
    print("Cross-Device Training Complete")
    print(f"Final accuracy: {history[-1]['accuracy']:.4f}")
    print(f"Avg participants: {avg_participants:.1f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
```

---

## Key Insights

### Cross-Device Best Practices

1. **Lightweight models**: Minimize on-device compute
2. **Smart selection**: Choose capable devices
3. **Handle dropouts**: Expect failures
4. **Compress updates**: Save bandwidth

---

## Exercises

1. **Exercise 1**: Add update compression
2. **Exercise 2**: Implement async training
3. **Exercise 3**: Design battery-aware scheduling
4. **Exercise 4**: Add differential privacy

---

## References

1. Bonawitz, K., et al. (2019). Towards FL at scale: A system design. In *MLSys*.
2. Hard, A., et al. (2019). FL for mobile keyboard prediction. *arXiv*.
3. Yang, T., et al. (2018). Applied FL. *AI Magazine*.

---

*Copyright 2026 Olaf Yunus Laitinen Imanov and Contributors. Released under EUPL 1.2.*
