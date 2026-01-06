# Tutorial 148: FL Case Study Mobile

---

## Metadata

| Property | Value |
|----------|-------|
| **Tutorial ID** | 148 |
| **Title** | Federated Learning Case Study: Mobile Devices |
| **Category** | Case Studies |
| **Difficulty** | Advanced |
| **Duration** | 120 minutes |
| **Prerequisites** | Tutorial 001-147 |
| **Author** | Unbitrium Contributors |
| **Last Updated** | January 2026 |

---

## Learning Objectives

By completing this tutorial, you will be able to:

1. **Understand** mobile FL applications and constraints
2. **Implement** keyboard prediction using FL
3. **Design** on-device learning systems
4. **Analyze** battery and bandwidth optimization
5. **Deploy** FL for mobile personalization

---

## Prerequisites

Before starting this tutorial, ensure you have:

- Completed tutorials 001-147
- Understanding of FL fundamentals
- Knowledge of mobile constraints
- Familiarity with NLP basics

---

## Background and Theory

### Mobile FL Challenges

Mobile devices present unique challenges for FL:
- Limited battery life
- Intermittent connectivity
- Variable compute capabilities
- Storage constraints
- User privacy expectations

### Mobile FL Architecture

```
Mobile FL Architecture:
├── Device Layer
│   ├── On-device training
│   ├── Model inference
│   ├── Local data storage
│   └── Battery management
├── Communication Layer
│   ├── Secure aggregation
│   ├── Compression
│   └── Opportunistic sync
└── Server Layer
    ├── Model aggregation
    ├── Client selection
    └── Model distribution
```

### Key Applications

| Application | Data Type | Update Frequency |
|-------------|-----------|------------------|
| Keyboard Prediction | Typing data | Daily |
| Voice Recognition | Audio samples | Weekly |
| App Recommendations | Usage patterns | Daily |
| Photo Organization | Image features | Weekly |

---

## Implementation Code

```python
#!/usr/bin/env python3
"""
Tutorial 148: Federated Learning Case Study - Mobile Devices

This module implements a federated learning system for mobile
applications, focusing on next-word prediction for keyboards.

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


class DeviceType(Enum):
    """Mobile device types with different capabilities."""
    HIGH_END = "high_end"      # Latest flagship
    MID_RANGE = "mid_range"    # Average device
    LOW_END = "low_end"        # Budget device


@dataclass
class MobileConfig:
    """Configuration for mobile FL simulation."""
    
    # FL parameters
    num_rounds: int = 50
    num_devices: int = 100
    devices_per_round: int = 20
    
    # Model parameters
    vocab_size: int = 1000
    embedding_dim: int = 64
    hidden_dim: int = 128
    seq_length: int = 10
    
    # Training parameters
    learning_rate: float = 0.01
    batch_size: int = 16
    local_epochs: int = 3
    
    # Mobile constraints
    max_model_size_mb: float = 10.0
    max_update_size_mb: float = 1.0
    min_battery_level: float = 0.3
    require_wifi: bool = True
    
    # Data parameters
    samples_per_device: int = 200
    
    seed: int = 42


class KeyboardDataset(Dataset):
    """Keyboard typing dataset for next-word prediction.
    
    Simulates user typing patterns with:
    - Word sequences
    - User-specific vocabulary preferences
    - Typing patterns
    """
    
    def __init__(
        self,
        user_id: int,
        vocab_size: int = 1000,
        n: int = 200,
        seq_length: int = 10,
        seed: int = 0
    ):
        np.random.seed(seed + user_id)
        
        self.user_id = user_id
        self.vocab_size = vocab_size
        self.n = n
        self.seq_length = seq_length
        
        # User-specific word preferences
        self.user_prefs = self._generate_user_preferences()
        
        # Generate sequences
        self.x, self.y = self._generate_sequences()
    
    def _generate_user_preferences(self) -> np.ndarray:
        """Generate user-specific word preferences."""
        # Each user has preferred words they use more often
        prefs = np.random.dirichlet(np.ones(self.vocab_size) * 0.1)
        return prefs
    
    def _generate_sequences(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate word sequences based on user preferences."""
        sequences = []
        targets = []
        
        for _ in range(self.n):
            # Sample words according to user preferences
            words = np.random.choice(
                self.vocab_size,
                size=self.seq_length + 1,
                p=self.user_prefs
            )
            
            sequences.append(words[:-1])
            targets.append(words[-1])
        
        x = torch.tensor(np.array(sequences), dtype=torch.long)
        y = torch.tensor(targets, dtype=torch.long)
        
        return x, y
    
    def __len__(self) -> int:
        return self.n
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx]


class NextWordPredictor(nn.Module):
    """LSTM-based next word prediction model.
    
    Designed for mobile deployment with size constraints.
    """
    
    def __init__(self, config: MobileConfig):
        super().__init__()
        
        self.config = config
        
        # Embedding layer
        self.embedding = nn.Embedding(
            config.vocab_size,
            config.embedding_dim
        )
        
        # LSTM for sequence processing
        self.lstm = nn.LSTM(
            input_size=config.embedding_dim,
            hidden_size=config.hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )
        
        # Output layer
        self.fc = nn.Linear(config.hidden_dim, config.vocab_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        # Embed words
        embedded = self.embedding(x)
        
        # Process sequence
        lstm_out, (h_n, c_n) = self.lstm(embedded)
        
        # Use final hidden state
        output = self.fc(h_n[-1])
        
        return output
    
    def get_model_size_mb(self) -> float:
        """Calculate model size in MB."""
        total_params = sum(
            p.numel() for p in self.parameters()
        )
        # Assuming float32 (4 bytes per param)
        return total_params * 4 / (1024 * 1024)
    
    def predict_next_words(
        self,
        sequence: torch.Tensor,
        top_k: int = 5
    ) -> List[Tuple[int, float]]:
        """Predict top-k next words."""
        self.eval()
        with torch.no_grad():
            output = self(sequence.unsqueeze(0))
            probs = F.softmax(output, dim=-1)
            values, indices = torch.topk(probs[0], top_k)
            
            return list(zip(indices.tolist(), values.tolist()))


class MobileDevice:
    """Simulated mobile device with constraints."""
    
    def __init__(
        self,
        device_id: int,
        dataset: KeyboardDataset,
        config: MobileConfig,
        device_type: DeviceType = DeviceType.MID_RANGE
    ):
        self.device_id = device_id
        self.dataset = dataset
        self.config = config
        self.device_type = device_type
        
        # Device state
        self.battery_level = np.random.uniform(0.2, 1.0)
        self.on_wifi = np.random.choice([True, False], p=[0.6, 0.4])
        self.is_charging = np.random.choice([True, False], p=[0.3, 0.7])
        
        # Device capabilities
        self._set_capabilities()
    
    def _set_capabilities(self) -> None:
        """Set device-specific capabilities."""
        if self.device_type == DeviceType.HIGH_END:
            self.compute_speed = 1.0
            self.memory_limit_mb = 256
        elif self.device_type == DeviceType.MID_RANGE:
            self.compute_speed = 0.6
            self.memory_limit_mb = 128
        else:
            self.compute_speed = 0.3
            self.memory_limit_mb = 64
    
    def is_eligible(self) -> bool:
        """Check if device is eligible for training."""
        # Battery check
        if self.battery_level < self.config.min_battery_level and not self.is_charging:
            return False
        
        # WiFi check
        if self.config.require_wifi and not self.on_wifi:
            return False
        
        return True
    
    def update_state(self) -> None:
        """Simulate device state changes."""
        # Battery drain/charge
        if self.is_charging:
            self.battery_level = min(1.0, self.battery_level + 0.1)
        else:
            self.battery_level = max(0.0, self.battery_level - 0.02)
        
        # WiFi changes
        self.on_wifi = np.random.choice([True, False], p=[0.6, 0.4])
    
    def train(self, model: nn.Module) -> Optional[Dict[str, Any]]:
        """Train on local data if eligible."""
        if not self.is_eligible():
            logger.debug(
                f"Device {self.device_id} not eligible "
                f"(battery={self.battery_level:.2f}, wifi={self.on_wifi})"
            )
            return None
        
        # Adjust epochs based on device type
        epochs = self.config.local_epochs
        if self.device_type == DeviceType.LOW_END:
            epochs = max(1, epochs // 2)
        
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
                output = local_model(x)
                loss = F.cross_entropy(output, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(local_model.parameters(), 1.0)
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
        
        # Simulate battery drain from training
        if not self.is_charging:
            self.battery_level -= 0.05
        
        return {
            "state_dict": {
                k: v.cpu() for k, v in local_model.state_dict().items()
            },
            "num_samples": len(self.dataset),
            "avg_loss": total_loss / num_batches,
            "device_id": self.device_id,
            "device_type": self.device_type.value
        }


class MobileUpdateCompressor:
    """Compress model updates for mobile bandwidth."""
    
    def __init__(self, compression_ratio: float = 0.1):
        self.compression_ratio = compression_ratio
    
    def compress(
        self,
        state_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, Any]:
        """Compress update using top-k sparsification."""
        compressed = {}
        
        for name, tensor in state_dict.items():
            flat = tensor.flatten()
            k = max(1, int(len(flat) * self.compression_ratio))
            
            values, indices = torch.topk(flat.abs(), k)
            selected_values = flat[indices]
            
            compressed[name] = {
                "indices": indices,
                "values": selected_values,
                "shape": tensor.shape
            }
        
        return compressed
    
    def decompress(
        self,
        compressed: Dict[str, Any]
    ) -> Dict[str, torch.Tensor]:
        """Decompress update."""
        state_dict = {}
        
        for name, data in compressed.items():
            flat = torch.zeros(int(np.prod(data["shape"])))
            flat[data["indices"]] = data["values"]
            state_dict[name] = flat.reshape(data["shape"])
        
        return state_dict


class MobileServer:
    """FL server for mobile devices."""
    
    def __init__(
        self,
        model: nn.Module,
        devices: List[MobileDevice],
        test_data: KeyboardDataset,
        config: MobileConfig
    ):
        self.model = model
        self.devices = devices
        self.test_data = test_data
        self.config = config
        
        self.compressor = MobileUpdateCompressor()
        self.history: List[Dict] = []
    
    def select_devices(self) -> List[MobileDevice]:
        """Select eligible devices for training."""
        eligible = [d for d in self.devices if d.is_eligible()]
        
        if len(eligible) == 0:
            logger.warning("No eligible devices!")
            return []
        
        n = min(self.config.devices_per_round, len(eligible))
        selected = np.random.choice(eligible, n, replace=False)
        
        return list(selected)
    
    def aggregate(self, updates: List[Dict]) -> None:
        """Aggregate device updates."""
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
        
        # Top-5 accuracy
        top5_correct = 0
        for x, y in loader:
            output = self.model(x)
            _, top5_pred = output.topk(5, dim=1)
            top5_correct += (top5_pred == y.unsqueeze(1)).any(dim=1).sum().item()
        
        return {
            "accuracy": correct / total,
            "top5_accuracy": top5_correct / total,
            "loss": total_loss / total
        }
    
    def train(self) -> List[Dict]:
        """Run FL training."""
        logger.info(f"Starting mobile FL with {len(self.devices)} devices")
        logger.info(f"Model size: {self.model.get_model_size_mb():.2f} MB")
        
        for round_num in range(self.config.num_rounds):
            # Update device states
            for device in self.devices:
                device.update_state()
            
            # Select devices
            selected = self.select_devices()
            
            if not selected:
                logger.warning(f"Round {round_num}: No devices available")
                continue
            
            # Collect updates
            updates = []
            for device in selected:
                update = device.train(self.model)
                if update is not None:
                    updates.append(update)
            
            if not updates:
                continue
            
            # Aggregate
            self.aggregate(updates)
            
            # Evaluate
            metrics = self.evaluate()
            
            record = {
                "round": round_num,
                **metrics,
                "num_devices": len(updates),
                "avg_loss": np.mean([u["avg_loss"] for u in updates])
            }
            self.history.append(record)
            
            if (round_num + 1) % 10 == 0:
                logger.info(
                    f"Round {round_num + 1}: "
                    f"acc={metrics['accuracy']:.4f}, "
                    f"top5={metrics['top5_accuracy']:.4f}, "
                    f"devices={len(updates)}"
                )
        
        return self.history


def main():
    """Main entry point."""
    print("=" * 60)
    print("Tutorial 148: FL Case Study - Mobile Devices")
    print("=" * 60)
    
    config = MobileConfig()
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    
    # Create device datasets
    device_types = [DeviceType.HIGH_END, DeviceType.MID_RANGE, DeviceType.LOW_END]
    devices = []
    
    for i in range(config.num_devices):
        dtype = device_types[i % 3]
        dataset = KeyboardDataset(
            user_id=i,
            vocab_size=config.vocab_size,
            n=config.samples_per_device,
            seq_length=config.seq_length,
            seed=config.seed
        )
        device = MobileDevice(i, dataset, config, dtype)
        devices.append(device)
    
    # Test data
    test_data = KeyboardDataset(
        user_id=999,
        vocab_size=config.vocab_size,
        n=500,
        seed=999
    )
    
    # Model
    model = NextWordPredictor(config)
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train
    server = MobileServer(model, devices, test_data, config)
    history = server.train()
    
    # Summary
    print("\n" + "=" * 60)
    print("Training Complete")
    print(f"Final Accuracy: {history[-1]['accuracy']:.4f}")
    print(f"Final Top-5 Accuracy: {history[-1]['top5_accuracy']:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
```

---

## Key Insights

### Mobile FL Challenges

1. **Device Heterogeneity**: Wide range of device capabilities
2. **Connectivity**: Intermittent and varying bandwidth
3. **Battery Constraints**: Training drains battery quickly
4. **Privacy**: Users expect on-device data to stay private

### Best Practices

- Check device eligibility before training
- Compress updates for bandwidth efficiency
- Adjust training based on device capabilities
- Use top-k accuracy for word prediction

---

## Exercises

1. **Exercise 1**: Add federated personalization layers
2. **Exercise 2**: Implement differential privacy
3. **Exercise 3**: Add model quantization for deployment
4. **Exercise 4**: Design bandwidth-aware client selection

---

## References

1. Hard, A., et al. (2018). Federated learning for mobile keyboard prediction. *arXiv*.
2. Bonawitz, K., et al. (2019). Towards FL at scale. In *MLSys*.
3. Yang, T., et al. (2018). Applied federated learning. *arXiv*.

---

*Copyright 2026 Olaf Yunus Laitinen Imanov and Contributors. Released under EUPL 1.2.*
