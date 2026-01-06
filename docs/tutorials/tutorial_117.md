# Tutorial 117: FL for IoT

---

## Metadata

| Property | Value |
|----------|-------|
| **Tutorial ID** | 117 |
| **Title** | Federated Learning for IoT Systems |
| **Category** | Domain Applications |
| **Difficulty** | Advanced |
| **Duration** | 120 minutes |
| **Prerequisites** | Tutorial 001-116 |
| **Author** | Unbitrium Contributors |
| **Last Updated** | January 2026 |

---

## Learning Objectives

By completing this tutorial, you will be able to:

1. **Understand** FL applications in IoT
2. **Implement** lightweight FL for embedded devices
3. **Design** hierarchical IoT FL architectures
4. **Analyze** resource constraints
5. **Deploy** FL on IoT platforms

---

## Prerequisites

Before starting this tutorial, ensure you have:

- Completed tutorials 001-116
- Understanding of FL fundamentals
- Knowledge of IoT systems
- Familiarity with embedded constraints

---

## Background and Theory

### IoT FL Challenges

IoT devices have unique constraints:
- Limited memory (KB to MB)
- Low compute power
- Battery operation
- Intermittent connectivity

### IoT FL Architecture

```
IoT FL Architecture:
├── Device Layer
│   ├── Sensors/Actuators
│   ├── Microcontrollers
│   └── Ultra-low power devices
├── Gateway Layer
│   ├── Edge aggregation
│   ├── Protocol translation
│   └── Local processing
├── Fog Layer
│   ├── Regional coordination
│   ├── Partial aggregation
│   └── Caching
└── Cloud Layer
    ├── Global aggregation
    ├── Model management
    └── Analytics
```

### IoT Device Classes

| Class | Memory | Compute | Example |
|-------|--------|---------|---------|
| Class 0 | <10KB | Minimal | Sensors |
| Class 1 | ~10KB | Low | Arduino |
| Class 2 | ~50KB | Low | ESP32 |
| Constrained | 256KB+ | Medium | RPi Zero |

---

## Implementation Code

```python
#!/usr/bin/env python3
"""
Tutorial 117: Federated Learning for IoT

This module implements FL for IoT with lightweight models,
hierarchical aggregation, and resource awareness.

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


class DeviceClass(Enum):
    """IoT device classifications."""
    CLASS_0 = "class_0"  # <10KB RAM
    CLASS_1 = "class_1"  # ~10KB RAM
    CLASS_2 = "class_2"  # ~50KB RAM
    CONSTRAINED = "constrained"  # 256KB+ RAM


@dataclass
class IoTConfig:
    """Configuration for IoT FL."""
    
    num_rounds: int = 50
    num_devices: int = 50
    devices_per_round: int = 20
    num_gateways: int = 5
    
    # Model parameters (tiny)
    input_dim: int = 10
    hidden_dim: int = 16
    num_classes: int = 5
    
    learning_rate: float = 0.01
    batch_size: int = 8
    local_epochs: int = 2
    
    # Gateway aggregation
    gateway_rounds: int = 3
    
    seed: int = 42


class IoTDataset(Dataset):
    """Sensor data dataset."""
    
    def __init__(
        self,
        device_id: int,
        n: int = 50,
        dim: int = 10,
        classes: int = 5,
        seed: int = 0
    ):
        np.random.seed(seed + device_id)
        
        # Simulate sensor readings
        self.x = torch.randn(n, dim, dtype=torch.float32) * 0.5
        self.y = torch.randint(0, classes, (n,), dtype=torch.long)
        
        # Add class-specific patterns
        for i in range(n):
            cls = self.y[i].item()
            self.x[i, cls % dim] += 1.0
    
    def __len__(self) -> int:
        return len(self.y)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx]


class TinyModel(nn.Module):
    """Ultra-lightweight model for IoT."""
    
    def __init__(self, config: IoTConfig):
        super().__init__()
        
        # Minimal architecture
        self.layer1 = nn.Linear(config.input_dim, config.hidden_dim)
        self.layer2 = nn.Linear(config.hidden_dim, config.num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.layer1(x))
        return self.layer2(x)
    
    def get_memory_bytes(self) -> int:
        """Estimate memory footprint."""
        params = sum(p.numel() * 4 for p in self.parameters())  # Float32
        activations = self.layer1.out_features * 4  # Estimated
        return params + activations
    
    def quantize_int8(self) -> Dict[str, torch.Tensor]:
        """Quantize to int8 for transmission."""
        quantized = {}
        for name, param in self.state_dict().items():
            scale = param.abs().max() / 127
            if scale > 0:
                q = (param / scale).round().clamp(-127, 127).to(torch.int8)
            else:
                q = torch.zeros_like(param, dtype=torch.int8)
            quantized[name] = {"data": q, "scale": scale.item()}
        return quantized
    
    @staticmethod
    def dequantize(quantized: Dict) -> Dict[str, torch.Tensor]:
        """Dequantize int8 to float32."""
        state = {}
        for name, q in quantized.items():
            state[name] = q["data"].float() * q["scale"]
        return state


class IoTDevice:
    """Simulated IoT device."""
    
    def __init__(
        self,
        device_id: int,
        dataset: IoTDataset,
        config: IoTConfig,
        device_class: DeviceClass = DeviceClass.CLASS_2,
        gateway_id: int = 0
    ):
        self.device_id = device_id
        self.dataset = dataset
        self.config = config
        self.device_class = device_class
        self.gateway_id = gateway_id
        
        # Battery simulation
        self.battery_level = np.random.uniform(0.5, 1.0)
    
    def can_train(self, model_bytes: int) -> bool:
        """Check if device can train."""
        # Memory check based on class
        class_limits = {
            DeviceClass.CLASS_0: 10 * 1024,
            DeviceClass.CLASS_1: 10 * 1024,
            DeviceClass.CLASS_2: 50 * 1024,
            DeviceClass.CONSTRAINED: 256 * 1024
        }
        
        if model_bytes > class_limits[self.device_class]:
            return False
        
        # Battery check
        if self.battery_level < 0.1:
            return False
        
        return True
    
    def train(self, model: nn.Module) -> Optional[Dict[str, Any]]:
        """Train on local sensor data."""
        model_bytes = model.get_memory_bytes()
        
        if not self.can_train(model_bytes):
            return None
        
        local = copy.deepcopy(model)
        optimizer = torch.optim.SGD(
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
            for x, y in loader:
                optimizer.zero_grad()
                loss = F.cross_entropy(local(x), y)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
        
        # Simulate battery drain
        self.battery_level -= 0.05
        
        # Quantize for transmission
        quantized = local.quantize_int8()
        
        return {
            "quantized": quantized,
            "num_samples": len(self.dataset),
            "avg_loss": total_loss / num_batches,
            "device_id": self.device_id,
            "gateway_id": self.gateway_id
        }


class IoTGateway:
    """Edge gateway for hierarchical aggregation."""
    
    def __init__(
        self,
        gateway_id: int,
        devices: List[IoTDevice],
        config: IoTConfig
    ):
        self.gateway_id = gateway_id
        self.devices = devices
        self.config = config
        
        # Local model at gateway
        self.local_model = TinyModel(config)
    
    def aggregate_local(
        self,
        updates: List[Dict]
    ) -> Dict[str, torch.Tensor]:
        """Aggregate updates from local devices."""
        if not updates:
            return {}
        
        total_samples = sum(u["num_samples"] for u in updates)
        new_state = {}
        
        for name in updates[0]["quantized"]:
            weighted_sum = None
            for u in updates:
                dequant = u["quantized"][name]["data"].float() * u["quantized"][name]["scale"]
                weight = u["num_samples"] / total_samples
                
                if weighted_sum is None:
                    weighted_sum = dequant * weight
                else:
                    weighted_sum += dequant * weight
            
            new_state[name] = weighted_sum
        
        return new_state
    
    def train_round(self, global_model: nn.Module) -> Dict[str, Any]:
        """Run local FL rounds."""
        self.local_model.load_state_dict(global_model.state_dict())
        
        for _ in range(self.config.gateway_rounds):
            # Collect from devices
            updates = []
            for device in self.devices:
                update = device.train(self.local_model)
                if update is not None:
                    updates.append(update)
            
            if not updates:
                continue
            
            # Aggregate locally
            aggregated = self.aggregate_local(updates)
            if aggregated:
                self.local_model.load_state_dict(aggregated)
        
        return {
            "state_dict": {k: v.cpu() for k, v in self.local_model.state_dict().items()},
            "num_samples": sum(len(d.dataset) for d in self.devices),
            "num_devices": len(self.devices),
            "gateway_id": self.gateway_id
        }


class IoTServer:
    """Cloud server for IoT FL."""
    
    def __init__(
        self,
        model: nn.Module,
        gateways: List[IoTGateway],
        test_data: IoTDataset,
        config: IoTConfig
    ):
        self.model = model
        self.gateways = gateways
        self.test_data = test_data
        self.config = config
        self.history: List[Dict] = []
    
    def aggregate(self, updates: List[Dict]) -> None:
        """Aggregate from gateways."""
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
        """Evaluate model."""
        self.model.eval()
        loader = DataLoader(self.test_data, batch_size=32)
        
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in loader:
                pred = self.model(x).argmax(dim=1)
                correct += (pred == y).sum().item()
                total += len(y)
        
        return {"accuracy": correct / total}
    
    def train(self) -> List[Dict]:
        """Run hierarchical FL."""
        logger.info(f"Starting IoT FL with {len(self.gateways)} gateways")
        
        for round_num in range(self.config.num_rounds):
            # Collect from gateways
            gateway_updates = []
            for gateway in self.gateways:
                update = gateway.train_round(self.model)
                gateway_updates.append(update)
            
            # Aggregate at cloud
            self.aggregate(gateway_updates)
            
            # Evaluate
            metrics = self.evaluate()
            
            total_devices = sum(u["num_devices"] for u in gateway_updates)
            
            record = {
                "round": round_num,
                **metrics,
                "num_gateways": len(gateway_updates),
                "total_devices": total_devices
            }
            self.history.append(record)
            
            if (round_num + 1) % 10 == 0:
                logger.info(
                    f"Round {round_num + 1}: "
                    f"acc={metrics['accuracy']:.4f}, "
                    f"devices={total_devices}"
                )
        
        return self.history


def main():
    """Main entry point."""
    print("=" * 60)
    print("Tutorial 117: FL for IoT")
    print("=" * 60)
    
    config = IoTConfig()
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    
    # Create devices and gateways
    device_classes = list(DeviceClass)
    gateways = []
    
    device_id = 0
    for gateway_id in range(config.num_gateways):
        devices = []
        devices_per_gateway = config.num_devices // config.num_gateways
        
        for _ in range(devices_per_gateway):
            device_class = device_classes[device_id % len(device_classes)]
            dataset = IoTDataset(
                device_id=device_id,
                n=50,
                dim=config.input_dim,
                seed=config.seed
            )
            device = IoTDevice(
                device_id=device_id,
                dataset=dataset,
                config=config,
                device_class=device_class,
                gateway_id=gateway_id
            )
            devices.append(device)
            device_id += 1
        
        gateway = IoTGateway(gateway_id, devices, config)
        gateways.append(gateway)
    
    # Test data
    test_data = IoTDataset(device_id=999, n=200, seed=999)
    
    # Model
    model = TinyModel(config)
    logger.info(f"Model size: {model.get_memory_bytes()} bytes")
    
    # Train
    server = IoTServer(model, gateways, test_data, config)
    history = server.train()
    
    print("\n" + "=" * 60)
    print("Training Complete")
    print(f"Final Accuracy: {history[-1]['accuracy']:.4f}")
    print(f"Model Size: {model.get_memory_bytes()} bytes")
    print("=" * 60)


if __name__ == "__main__":
    main()
```

---

## Key Insights

### IoT FL Best Practices

1. **Ultra-lightweight models**: KB-sized
2. **Hierarchical aggregation**: Via gateways
3. **Quantization**: Int8 for transmission
4. **Battery awareness**: Preserve energy

---

## Exercises

1. **Exercise 1**: Implement duty cycling
2. **Exercise 2**: Add adaptive model complexity
3. **Exercise 3**: Design sleep scheduling
4. **Exercise 4**: Implement over-the-air updates

---

## References

1. Lim, W.Y.B., et al. (2020). FL in mobile edge networks. *IEEE COMST*.
2. Khan, L.U., et al. (2021). FL for IoT networks. *IEEE IoT Journal*.
3. Saha, A., et al. (2020). Federated transfer learning for IoT. *IEEE TMC*.

---

*Copyright 2026 Olaf Yunus Laitinen Imanov and Contributors. Released under EUPL 1.2.*
