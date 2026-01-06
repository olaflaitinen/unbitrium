# Tutorial 118: FL for Autonomous Vehicles

---

## Metadata

| Property | Value |
|----------|-------|
| **Tutorial ID** | 118 |
| **Title** | Federated Learning for Autonomous Vehicles |
| **Category** | Domain Applications |
| **Difficulty** | Expert |
| **Duration** | 120 minutes |
| **Prerequisites** | Tutorial 001-117 |
| **Author** | Unbitrium Contributors |
| **Last Updated** | January 2026 |

---

## Learning Objectives

By completing this tutorial, you will be able to:

1. **Understand** FL applications in autonomous driving
2. **Implement** federated perception models
3. **Design** vehicle-to-cloud FL systems
4. **Analyze** safety-critical requirements
5. **Deploy** real-time FL for vehicles

---

## Prerequisites

Before starting this tutorial, ensure you have:

- Completed tutorials 001-117
- Understanding of FL fundamentals
- Knowledge of autonomous vehicle systems
- Familiarity with perception models

---

## Background and Theory

### AV FL Challenges

Autonomous vehicles present unique FL challenges:
- Safety-critical decisions
- Real-time requirements
- Diverse driving conditions
- Privacy of driving patterns

### Architecture

```
AV FL Architecture:
├── Vehicle Layer
│   ├── Perception (cameras, LiDAR)
│   ├── Local model inference
│   ├── Data collection
│   └── On-vehicle training
├── Edge Layer (RSU)
│   ├── Regional aggregation
│   ├── Low-latency updates
│   └── Traffic coordination
└── Cloud Layer
    ├── Global model management
    ├── Fleet-wide learning
    └── Model validation
```

### Safety Requirements

| Requirement | Description | FL Impact |
|-------------|-------------|-----------|
| Real-time | <100ms inference | Edge deployment |
| Reliability | 99.99% uptime | Fault tolerance |
| Validation | Extensive testing | Pre-deployment |
| Explainability | Decision auditing | Model transparency |

---

## Implementation Code

```python
#!/usr/bin/env python3
"""
Tutorial 118: FL for Autonomous Vehicles

This module implements federated learning for
autonomous vehicle perception systems.

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
from enum import Enum
import copy
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DrivingCondition(Enum):
    """Driving environment conditions."""
    URBAN = "urban"
    HIGHWAY = "highway"
    RURAL = "rural"
    NIGHT = "night"
    RAIN = "rain"
    SNOW = "snow"


@dataclass
class AVConfig:
    """Configuration for AV FL."""
    
    num_rounds: int = 50
    num_vehicles: int = 20
    vehicles_per_round: int = 10
    
    # Perception model
    input_channels: int = 3
    image_size: int = 32
    hidden_dim: int = 64
    num_classes: int = 5  # Vehicle, pedestrian, cyclist, traffic sign, other
    
    learning_rate: float = 0.01
    batch_size: int = 16
    local_epochs: int = 2
    
    seed: int = 42


class DrivingDataset(Dataset):
    """Simulated driving perception dataset."""
    
    def __init__(
        self,
        vehicle_id: int,
        condition: DrivingCondition,
        n: int = 200,
        img_size: int = 32,
        num_classes: int = 5,
        seed: int = 0
    ):
        np.random.seed(seed + vehicle_id)
        
        self.condition = condition
        
        # Simulated images (3 channels)
        self.images = torch.randn(n, 3, img_size, img_size)
        self.labels = torch.randint(0, num_classes, (n,))
        
        # Add condition-specific patterns
        self._add_condition_bias()
    
    def _add_condition_bias(self):
        """Add condition-specific patterns."""
        if self.condition == DrivingCondition.NIGHT:
            self.images *= 0.5  # Darker
        elif self.condition == DrivingCondition.RAIN:
            noise = torch.randn_like(self.images) * 0.3
            self.images += noise
        elif self.condition == DrivingCondition.SNOW:
            self.images += 0.2  # Brighter
    
    def __len__(self) -> int:
        return len(self.labels)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.images[idx], self.labels[idx]


class PerceptionModel(nn.Module):
    """Simplified perception model for object detection."""
    
    def __init__(self, config: AVConfig):
        super().__init__()
        
        # CNN backbone
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        
        # Calculate flattened size
        size = config.image_size // 4  # After 2 pools
        self.fc_input = 32 * size * size
        
        self.fc1 = nn.Linear(self.fc_input, config.hidden_dim)
        self.fc2 = nn.Linear(config.hidden_dim, config.num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class VehicleClient:
    """Vehicle as FL client."""
    
    def __init__(
        self,
        vehicle_id: int,
        dataset: DrivingDataset,
        config: AVConfig,
        condition: DrivingCondition
    ):
        self.vehicle_id = vehicle_id
        self.dataset = dataset
        self.config = config
        self.condition = condition
        
        # Vehicle state
        self.is_connected = True
        self.battery_level = 1.0
    
    def can_train(self) -> bool:
        """Check if vehicle can participate."""
        return self.is_connected and self.battery_level > 0.2
    
    def train(self, model: nn.Module) -> Optional[Dict[str, Any]]:
        """Train on local driving data."""
        if not self.can_train():
            return None
        
        local = copy.deepcopy(model)
        optimizer = torch.optim.Adam(
            local.parameters(),
            lr=self.config.learning_rate
        )
        
        loader = DataLoader(
            self.dataset,
            batch_size=self.config.batch_size,
            shuffle=True
        )
        
        local.train()
        total_loss = 0.0
        num_batches = 0
        
        for _ in range(self.config.local_epochs):
            for images, labels in loader:
                optimizer.zero_grad()
                output = local(images)
                loss = F.cross_entropy(output, labels)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
        
        # Simulate battery drain
        self.battery_level -= 0.05
        
        return {
            "state_dict": {k: v.cpu() for k, v in local.state_dict().items()},
            "num_samples": len(self.dataset),
            "avg_loss": total_loss / num_batches,
            "vehicle_id": self.vehicle_id,
            "condition": self.condition.value
        }


class EdgeRSU:
    """Road Side Unit for edge aggregation."""
    
    def __init__(
        self,
        rsu_id: int,
        vehicles: List[VehicleClient],
        config: AVConfig
    ):
        self.rsu_id = rsu_id
        self.vehicles = vehicles
        self.config = config
        self.local_model = PerceptionModel(config)
    
    def aggregate_local(self, updates: List[Dict]) -> Dict[str, torch.Tensor]:
        """Aggregate updates from nearby vehicles."""
        if not updates:
            return {}
        
        total_samples = sum(u["num_samples"] for u in updates)
        new_state = {}
        
        for key in updates[0]["state_dict"]:
            new_state[key] = sum(
                (u["num_samples"] / total_samples) * u["state_dict"][key].float()
                for u in updates
            )
        
        self.local_model.load_state_dict(new_state)
        return new_state
    
    def collect_and_aggregate(
        self,
        global_model: nn.Module
    ) -> Dict[str, Any]:
        """Collect from vehicles and aggregate."""
        updates = []
        conditions = {}
        
        for vehicle in self.vehicles:
            if vehicle.can_train():
                update = vehicle.train(global_model)
                if update:
                    updates.append(update)
                    cond = update["condition"]
                    conditions[cond] = conditions.get(cond, 0) + 1
        
        if updates:
            aggregated = self.aggregate_local(updates)
        else:
            aggregated = global_model.state_dict()
        
        return {
            "state_dict": aggregated,
            "num_vehicles": len(updates),
            "condition_distribution": conditions,
            "rsu_id": self.rsu_id
        }


class AVFLServer:
    """Cloud server for AV fleet learning."""
    
    def __init__(
        self,
        model: nn.Module,
        rsus: List[EdgeRSU],
        test_data: DrivingDataset,
        config: AVConfig
    ):
        self.model = model
        self.rsus = rsus
        self.test_data = test_data
        self.config = config
        self.history: List[Dict] = []
    
    def aggregate(self, rsu_updates: List[Dict]) -> None:
        """Aggregate from RSUs."""
        if not rsu_updates:
            return
        
        total_vehicles = sum(u["num_vehicles"] for u in rsu_updates)
        if total_vehicles == 0:
            return
        
        new_state = {}
        for key in rsu_updates[0]["state_dict"]:
            new_state[key] = sum(
                (u["num_vehicles"] / total_vehicles) * u["state_dict"][key].float()
                for u in rsu_updates
                if u["state_dict"]
            )
        
        self.model.load_state_dict(new_state)
    
    def evaluate(self) -> Dict[str, float]:
        """Evaluate perception model."""
        self.model.eval()
        loader = DataLoader(self.test_data, batch_size=32)
        
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in loader:
                pred = self.model(images).argmax(dim=1)
                correct += (pred == labels).sum().item()
                total += len(labels)
        
        return {"accuracy": correct / total}
    
    def train(self) -> List[Dict]:
        """Run hierarchical FL for AV fleet."""
        logger.info(f"Starting AV FL with {len(self.rsus)} RSUs")
        
        for round_num in range(self.config.num_rounds):
            # Collect from all RSUs
            rsu_updates = []
            total_vehicles = 0
            
            for rsu in self.rsus:
                update = rsu.collect_and_aggregate(self.model)
                rsu_updates.append(update)
                total_vehicles += update["num_vehicles"]
            
            # Global aggregation
            self.aggregate(rsu_updates)
            
            # Evaluate
            metrics = self.evaluate()
            
            record = {
                "round": round_num,
                **metrics,
                "num_vehicles": total_vehicles,
                "num_rsus": len(rsu_updates)
            }
            self.history.append(record)
            
            if (round_num + 1) % 10 == 0:
                logger.info(
                    f"Round {round_num + 1}: "
                    f"acc={metrics['accuracy']:.4f}, "
                    f"vehicles={total_vehicles}"
                )
        
        return self.history


def main():
    """Main entry point."""
    print("=" * 60)
    print("Tutorial 118: FL for Autonomous Vehicles")
    print("=" * 60)
    
    config = AVConfig()
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    
    # Create vehicles with different conditions
    conditions = list(DrivingCondition)
    num_rsus = 4
    vehicles_per_rsu = config.num_vehicles // num_rsus
    
    rsus = []
    vehicle_id = 0
    
    for rsu_id in range(num_rsus):
        vehicles = []
        for _ in range(vehicles_per_rsu):
            condition = conditions[vehicle_id % len(conditions)]
            dataset = DrivingDataset(
                vehicle_id=vehicle_id,
                condition=condition,
                img_size=config.image_size,
                seed=config.seed
            )
            vehicle = VehicleClient(vehicle_id, dataset, config, condition)
            vehicles.append(vehicle)
            vehicle_id += 1
        
        rsu = EdgeRSU(rsu_id, vehicles, config)
        rsus.append(rsu)
    
    # Test data
    test_data = DrivingDataset(
        vehicle_id=999,
        condition=DrivingCondition.URBAN,
        n=300,
        seed=999
    )
    
    # Model
    model = PerceptionModel(config)
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train
    server = AVFLServer(model, rsus, test_data, config)
    history = server.train()
    
    print("\n" + "=" * 60)
    print("Training Complete")
    print(f"Final Accuracy: {history[-1]['accuracy']:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
```

---

## Key Insights

### AV FL Considerations

1. **Safety first**: Validate before deployment
2. **Hierarchical**: Use RSUs for edge aggregation
3. **Condition diversity**: Handle varied environments
4. **Real-time**: Low-latency requirements

---

## Exercises

1. **Exercise 1**: Add multi-task learning
2. **Exercise 2**: Implement V2V communication
3. **Exercise 3**: Add safety validation
4. **Exercise 4**: Design domain adaptation

---

## References

1. Elbir, A.M., et al. (2021). FL for vehicular networks. *IEEE VT Magazine*.
2. Du, Z., et al. (2020). FL for autonomous driving. *IEEE Access*.
3. Pokhrel, S.R., et al. (2020). FL for connected vehicles. *IEEE IoT Journal*.

---

*Copyright 2026 Olaf Yunus Laitinen Imanov and Contributors. Released under EUPL 1.2.*
