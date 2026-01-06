# Tutorial 136: FL Edge Computing

---

## Metadata

| Property | Value |
|----------|-------|
| **Tutorial ID** | 136 |
| **Title** | Federated Learning for Edge Computing |
| **Category** | Infrastructure |
| **Difficulty** | Advanced |
| **Duration** | 120 minutes |
| **Prerequisites** | Tutorial 001-135 |
| **Author** | Unbitrium Contributors |
| **Last Updated** | January 2026 |

---

## Learning Objectives

By completing this tutorial, you will be able to:

1. **Understand** edge computing FL architecture
2. **Implement** hierarchical FL with edge aggregation
3. **Design** resource-constrained edge training
4. **Analyze** edge-cloud FL coordination
5. **Deploy** FL models on edge devices

---

## Prerequisites

Before starting this tutorial, ensure you have:

- Completed tutorials 001-135
- Understanding of FL fundamentals
- Knowledge of distributed systems
- Familiarity with edge computing concepts

---

## Background and Theory

### Edge Computing and FL

Edge computing brings computation closer to data sources:
- Reduced latency
- Bandwidth savings
- Privacy preservation
- Real-time processing

FL is naturally suited for edge computing due to:
- Local data processing
- Model compression capabilities
- Hierarchical aggregation support
- Privacy by design

### Edge FL Architecture

```
Edge FL Architecture:
├── Device Tier
│   ├── IoT sensors
│   ├── Mobile devices
│   └── Embedded systems
├── Edge Tier
│   ├── Edge servers
│   ├── Local aggregation
│   └── Model caching
├── Fog Tier
│   ├── Regional coordinators
│   ├── Partial aggregation
│   └── Resource scheduling
└── Cloud Tier
    ├── Global aggregation
    ├── Model management
    └── Analytics
```

### Resource Constraints

| Device Type | Memory | Compute | Network |
|-------------|--------|---------|---------|
| Microcontroller | <256KB | Very Low | LoRa/BLE |
| Raspberry Pi | 1-8GB | Low | WiFi |
| Edge Server | 16-64GB | Medium | 1Gbps |
| Cloud | Unlimited | High | 10Gbps+ |

---

## Implementation Code

```python
#!/usr/bin/env python3
"""
Tutorial 136: Federated Learning for Edge Computing

This module implements hierarchical FL with edge aggregation
for resource-constrained edge device environments.

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


class DeviceTier(Enum):
    """Device capability tiers."""
    IOT = "iot"           # Microcontrollers
    MOBILE = "mobile"     # Phones, tablets
    EDGE = "edge"         # Edge servers
    CLOUD = "cloud"       # Cloud servers


@dataclass
class EdgeConfig:
    """Configuration for edge FL."""

    # FL parameters
    num_rounds: int = 50
    num_edge_nodes: int = 5      # Edge aggregators
    devices_per_edge: int = 10   # Devices per edge

    # Model parameters
    input_dim: int = 32
    hidden_dim: int = 64
    num_classes: int = 10

    # Training parameters
    learning_rate: float = 0.01
    batch_size: int = 16
    local_epochs: int = 3
    edge_aggregation_rounds: int = 3  # Local rounds before cloud sync

    # Resource constraints
    max_model_size_kb: float = 500
    quantization_bits: int = 8

    # Data parameters
    samples_per_device: int = 100

    seed: int = 42


class EdgeDataset(Dataset):
    """Dataset for edge device."""

    def __init__(
        self,
        device_id: int,
        n: int = 100,
        dim: int = 32,
        classes: int = 10,
        seed: int = 0
    ):
        np.random.seed(seed + device_id)

        self.x = torch.randn(n, dim, dtype=torch.float32)
        self.y = torch.randint(0, classes, (n,), dtype=torch.long)

        # Add class patterns
        for i in range(n):
            self.x[i, self.y[i].item() % dim] += 2.0

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx]


class LightweightModel(nn.Module):
    """Lightweight model for edge deployment."""

    def __init__(self, config: EdgeConfig):
        super().__init__()

        self.config = config

        # Compact architecture
        self.layers = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(config.hidden_dim // 2, config.num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)

    def get_model_size_kb(self) -> float:
        """Calculate model size in KB."""
        total_params = sum(p.numel() for p in self.parameters())
        # Float32 = 4 bytes
        return total_params * 4 / 1024

    def quantize(self, bits: int = 8) -> Dict[str, torch.Tensor]:
        """Quantize model for transmission."""
        quantized = {}
        max_val = 2 ** (bits - 1) - 1

        for name, param in self.state_dict().items():
            scale = param.abs().max() / max_val
            if scale > 0:
                q = torch.round(param / scale).to(torch.int8)
            else:
                q = torch.zeros_like(param, dtype=torch.int8)
            quantized[name] = (q, scale)

        return quantized

    def dequantize(self, quantized: Dict[str, Tuple]) -> None:
        """Load quantized weights."""
        state_dict = {}
        for name, (q, scale) in quantized.items():
            state_dict[name] = q.float() * scale
        self.load_state_dict(state_dict)


class EdgeDevice:
    """Simulated edge device with resource constraints."""

    def __init__(
        self,
        device_id: int,
        dataset: EdgeDataset,
        config: EdgeConfig,
        tier: DeviceTier = DeviceTier.IOT
    ):
        self.device_id = device_id
        self.dataset = dataset
        self.config = config
        self.tier = tier

        # Set capabilities based on tier
        self._set_capabilities()

    def _set_capabilities(self) -> None:
        """Set device capabilities based on tier."""
        if self.tier == DeviceTier.IOT:
            self.memory_kb = 64
            self.compute_factor = 0.2
            self.can_train = True
        elif self.tier == DeviceTier.MOBILE:
            self.memory_kb = 512
            self.compute_factor = 0.5
            self.can_train = True
        else:
            self.memory_kb = 4096
            self.compute_factor = 1.0
            self.can_train = True

    def train(self, model: nn.Module) -> Optional[Dict[str, Any]]:
        """Train on local data with resource constraints."""

        if not self.can_train:
            return None

        # Adjust epochs based on compute capability
        epochs = max(1, int(self.config.local_epochs * self.compute_factor))

        local_model = copy.deepcopy(model)
        optimizer = torch.optim.SGD(
            local_model.parameters(),
            lr=self.config.learning_rate
        )

        loader = DataLoader(
            self.dataset,
            batch_size=min(self.config.batch_size, len(self.dataset)),
            shuffle=True
        )

        local_model.train()
        total_loss = 0.0
        num_batches = 0

        for _ in range(epochs):
            for x, y in loader:
                optimizer.zero_grad()
                output = local_model(x)
                loss = F.cross_entropy(output, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(local_model.parameters(), 1.0)
                optimizer.step()

                total_loss += loss.item()
                num_batches += 1

        # Quantize for transmission
        quantized = local_model.quantize(self.config.quantization_bits)

        return {
            "quantized": quantized,
            "num_samples": len(self.dataset),
            "avg_loss": total_loss / num_batches,
            "device_id": self.device_id,
            "tier": self.tier.value
        }


class EdgeAggregator:
    """Edge server that aggregates from local devices."""

    def __init__(
        self,
        edge_id: int,
        devices: List[EdgeDevice],
        config: EdgeConfig
    ):
        self.edge_id = edge_id
        self.devices = devices
        self.config = config

        # Local model at edge
        self.local_model = LightweightModel(config)

    def aggregate_local(
        self,
        updates: List[Dict]
    ) -> Dict[str, torch.Tensor]:
        """Aggregate updates from local devices."""

        if not updates:
            return {}

        total_samples = sum(u["num_samples"] for u in updates)
        new_state = {}

        # First update to get keys
        first_quantized = updates[0]["quantized"]

        for key in first_quantized:
            # Dequantize and weight
            weighted_sum = None
            for u in updates:
                q, scale = u["quantized"][key]
                dequant = q.float() * scale
                weight = u["num_samples"] / total_samples

                if weighted_sum is None:
                    weighted_sum = dequant * weight
                else:
                    weighted_sum += dequant * weight

            new_state[key] = weighted_sum

        return new_state

    def train_round(self, global_model: nn.Module) -> Dict[str, Any]:
        """Run local training round."""

        # Distribute global model to devices
        self.local_model.load_state_dict(global_model.state_dict())

        # Collect updates from devices
        updates = []
        for device in self.devices:
            update = device.train(self.local_model)
            if update is not None:
                updates.append(update)

        if not updates:
            return {
                "status": "no_updates",
                "edge_id": self.edge_id
            }

        # Aggregate locally
        aggregated_state = self.aggregate_local(updates)
        self.local_model.load_state_dict(aggregated_state)

        return {
            "state_dict": aggregated_state,
            "num_samples": sum(u["num_samples"] for u in updates),
            "avg_loss": np.mean([u["avg_loss"] for u in updates]),
            "num_devices": len(updates),
            "edge_id": self.edge_id,
            "status": "success"
        }


class CloudServer:
    """Cloud server for global aggregation."""

    def __init__(
        self,
        model: nn.Module,
        edge_aggregators: List[EdgeAggregator],
        test_data: EdgeDataset,
        config: EdgeConfig
    ):
        self.model = model
        self.edge_aggregators = edge_aggregators
        self.test_data = test_data
        self.config = config
        self.history: List[Dict] = []

    def aggregate_edges(
        self,
        edge_updates: List[Dict]
    ) -> None:
        """Aggregate from edge servers."""

        successful = [u for u in edge_updates if u.get("status") == "success"]

        if not successful:
            logger.warning("No successful edge updates")
            return

        total_samples = sum(u["num_samples"] for u in successful)
        new_state = {}

        for key in successful[0]["state_dict"]:
            new_state[key] = sum(
                (u["num_samples"] / total_samples) * u["state_dict"][key].float()
                for u in successful
            )

        self.model.load_state_dict(new_state)

    def evaluate(self) -> Dict[str, float]:
        """Evaluate global model."""
        self.model.eval()
        loader = DataLoader(self.test_data, batch_size=64)

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

    def train(self) -> List[Dict]:
        """Run hierarchical FL training."""
        logger.info(
            f"Starting hierarchical FL with "
            f"{len(self.edge_aggregators)} edge servers"
        )

        for round_num in range(self.config.num_rounds):
            # Run local aggregation at each edge
            edge_updates = []
            for edge in self.edge_aggregators:
                # Run multiple local rounds before cloud sync
                for _ in range(self.config.edge_aggregation_rounds):
                    update = edge.train_round(self.model)
                edge_updates.append(update)

            # Aggregate at cloud
            self.aggregate_edges(edge_updates)

            # Evaluate
            metrics = self.evaluate()

            successful_edges = len([
                u for u in edge_updates if u.get("status") == "success"
            ])
            total_devices = sum(
                u.get("num_devices", 0) for u in edge_updates
            )

            record = {
                "round": round_num,
                **metrics,
                "num_edges": successful_edges,
                "total_devices": total_devices
            }
            self.history.append(record)

            if (round_num + 1) % 10 == 0:
                logger.info(
                    f"Round {round_num + 1}: "
                    f"acc={metrics['accuracy']:.4f}, "
                    f"edges={successful_edges}, "
                    f"devices={total_devices}"
                )

        return self.history


def main():
    """Main entry point."""
    print("=" * 60)
    print("Tutorial 136: FL for Edge Computing")
    print("=" * 60)

    config = EdgeConfig()
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    # Create edge aggregators with devices
    device_tiers = [DeviceTier.IOT, DeviceTier.MOBILE, DeviceTier.EDGE]
    edge_aggregators = []

    device_counter = 0
    for edge_id in range(config.num_edge_nodes):
        devices = []
        for _ in range(config.devices_per_edge):
            tier = device_tiers[device_counter % 3]
            dataset = EdgeDataset(
                device_id=device_counter,
                n=config.samples_per_device,
                dim=config.input_dim,
                seed=config.seed
            )
            device = EdgeDevice(device_counter, dataset, config, tier)
            devices.append(device)
            device_counter += 1

        aggregator = EdgeAggregator(edge_id, devices, config)
        edge_aggregators.append(aggregator)

    # Test data
    test_data = EdgeDataset(
        device_id=999,
        n=500,
        seed=999
    )

    # Model
    model = LightweightModel(config)
    logger.info(f"Model size: {model.get_model_size_kb():.2f} KB")
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Train
    server = CloudServer(model, edge_aggregators, test_data, config)
    history = server.train()

    # Summary
    print("\n" + "=" * 60)
    print("Training Complete")
    print(f"Final Accuracy: {history[-1]['accuracy']:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
```

---

## Key Insights

### Edge FL Challenges

1. **Resource Heterogeneity**: Wide range of device capabilities
2. **Hierarchical Aggregation**: Multi-tier aggregation needed
3. **Model Size**: Must fit in constrained memory
4. **Communication**: Limited and expensive

### Best Practices

- Use quantization for communication efficiency
- Implement hierarchical aggregation
- Adapt training to device capabilities
- Design lightweight models

---

## Exercises

1. **Exercise 1**: Add bandwidth simulation
2. **Exercise 2**: Implement pruning for smaller models
3. **Exercise 3**: Add device failure handling
4. **Exercise 4**: Design adaptive local epochs

---

## References

1. Wang, S., et al. (2019). Adaptive FL in resource constrained edge computing. *IEEE TMC*.
2. Diao, E., et al. (2021). HeteroFL: Computation and communication efficient FL for heterogeneous clients. In *ICLR*.
3. Lin, T., et al. (2020). Communication-efficient FL at the network edge. In *INFOCOM*.

---

*Copyright 2026 Olaf Yunus Laitinen Imanov and Contributors. Released under EUPL 1.2.*
