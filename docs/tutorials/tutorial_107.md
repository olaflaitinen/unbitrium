# Tutorial 107: FL Resource Management

---

## Metadata

| Property | Value |
|----------|-------|
| **Tutorial ID** | 107 |
| **Title** | Federated Learning Resource Management |
| **Category** | Systems |
| **Difficulty** | Advanced |
| **Duration** | 120 minutes |
| **Prerequisites** | Tutorial 001-106 |
| **Author** | Unbitrium Contributors |
| **Last Updated** | January 2026 |

---

## Learning Objectives

By completing this tutorial, you will be able to:

1. **Understand** resource constraints in FL
2. **Implement** adaptive resource allocation
3. **Design** efficient client scheduling
4. **Analyze** system utilization
5. **Deploy** resource-aware FL systems

---

## Prerequisites

Before starting this tutorial, ensure you have:

- Completed tutorials 001-106
- Understanding of FL fundamentals
- Knowledge of system resources
- Familiarity with scheduling algorithms

---

## Background and Theory

### Resource Types in FL

```
FL Resources:
├── Compute
│   ├── CPU/GPU cycles
│   ├── Memory bandwidth
│   └── Cache efficiency
├── Memory
│   ├── Model storage
│   ├── Activation memory
│   └── Gradient buffers
├── Network
│   ├── Bandwidth
│   ├── Latency
│   └── Reliability
└── Energy
    ├── Battery life
    ├── Power budget
    └── Thermal limits
```

### Resource Management Goals

| Goal | Description | Approach |
|------|-------------|----------|
| Efficiency | Max throughput | Load balancing |
| Fairness | Equal participation | Round-robin |
| Latency | Min round time | Async |
| Energy | Min consumption | Adaptive |

---

## Implementation Code

```python
#!/usr/bin/env python3
"""
Tutorial 107: Federated Learning Resource Management

This module implements resource-aware FL with adaptive
allocation and efficient scheduling.

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
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ResourceConfig:
    """Configuration for resource-aware FL."""
    
    num_rounds: int = 50
    num_clients: int = 20
    clients_per_round: int = 10
    
    input_dim: int = 32
    hidden_dim: int = 64
    num_classes: int = 10
    
    learning_rate: float = 0.01
    batch_size: int = 32
    local_epochs: int = 3
    
    # Resource parameters
    max_memory_mb: float = 512
    max_bandwidth_mbps: float = 100
    target_round_time: float = 10.0  # seconds
    
    seed: int = 42


@dataclass
class ClientResources:
    """Resource profile for a client."""
    compute_speed: float = 1.0  # Relative speed
    memory_mb: float = 256
    bandwidth_mbps: float = 50
    battery_level: float = 1.0
    is_charging: bool = False


class ResourceMonitor:
    """Monitor and estimate resource usage."""
    
    def __init__(self, num_clients: int):
        self.num_clients = num_clients
        self.profiles: Dict[int, ClientResources] = {}
        self.history: Dict[int, List[Dict]] = {i: [] for i in range(num_clients)}
    
    def register_profile(
        self,
        client_id: int,
        resources: ClientResources
    ) -> None:
        """Register client resource profile."""
        self.profiles[client_id] = resources
    
    def record_usage(
        self,
        client_id: int,
        metrics: Dict[str, float]
    ) -> None:
        """Record resource usage."""
        self.history[client_id].append(metrics)
    
    def estimate_time(
        self,
        client_id: int,
        data_size: int,
        model_size_mb: float
    ) -> float:
        """Estimate training time for client."""
        if client_id not in self.profiles:
            return float('inf')
        
        profile = self.profiles[client_id]
        
        # Compute time estimate
        compute_time = data_size / (1000 * profile.compute_speed)
        
        # Communication time
        comm_time = model_size_mb / profile.bandwidth_mbps
        
        return compute_time + comm_time
    
    def get_available_clients(
        self,
        min_battery: float = 0.2,
        min_memory: float = 100
    ) -> List[int]:
        """Get clients with sufficient resources."""
        available = []
        for cid, profile in self.profiles.items():
            if profile.battery_level >= min_battery or profile.is_charging:
                if profile.memory_mb >= min_memory:
                    available.append(cid)
        return available


class ResourceScheduler:
    """Schedule clients based on resources."""
    
    def __init__(
        self,
        monitor: ResourceMonitor,
        config: ResourceConfig
    ):
        self.monitor = monitor
        self.config = config
        self.participation_count: Dict[int, int] = {}
    
    def select_clients(
        self,
        available: List[int],
        n: int,
        strategy: str = "fastest"
    ) -> List[int]:
        """Select clients using specified strategy."""
        if strategy == "random":
            return self._random_select(available, n)
        elif strategy == "fastest":
            return self._fastest_select(available, n)
        elif strategy == "fair":
            return self._fair_select(available, n)
        elif strategy == "energy_aware":
            return self._energy_aware_select(available, n)
        else:
            return available[:n]
    
    def _random_select(self, available: List[int], n: int) -> List[int]:
        """Random selection."""
        return list(np.random.choice(available, min(n, len(available)), replace=False))
    
    def _fastest_select(self, available: List[int], n: int) -> List[int]:
        """Select fastest clients."""
        times = [(cid, self.monitor.estimate_time(cid, 100, 1.0)) for cid in available]
        times.sort(key=lambda x: x[1])
        return [cid for cid, _ in times[:n]]
    
    def _fair_select(self, available: List[int], n: int) -> List[int]:
        """Select least-participated clients."""
        counts = [(cid, self.participation_count.get(cid, 0)) for cid in available]
        counts.sort(key=lambda x: x[1])
        selected = [cid for cid, _ in counts[:n]]
        
        for cid in selected:
            self.participation_count[cid] = self.participation_count.get(cid, 0) + 1
        
        return selected
    
    def _energy_aware_select(self, available: List[int], n: int) -> List[int]:
        """Select clients with good energy status."""
        def energy_score(cid):
            profile = self.monitor.profiles.get(cid)
            if profile is None:
                return 0
            if profile.is_charging:
                return 2.0
            return profile.battery_level
        
        scores = [(cid, energy_score(cid)) for cid in available]
        scores.sort(key=lambda x: x[1], reverse=True)
        return [cid for cid, _ in scores[:n]]


class ResourceDataset(Dataset):
    """Dataset for resource experiments."""
    
    def __init__(
        self,
        client_id: int,
        n: int = 200,
        dim: int = 32,
        classes: int = 10,
        seed: int = 0
    ):
        np.random.seed(seed + client_id)
        
        self.x = torch.randn(n, dim, dtype=torch.float32)
        self.y = torch.randint(0, classes, (n,), dtype=torch.long)
        
        for i in range(n):
            self.x[i, self.y[i].item() % dim] += 2.0
    
    def __len__(self) -> int:
        return len(self.y)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx]


class ResourceModel(nn.Module):
    """Model with size metrics."""
    
    def __init__(self, config: ResourceConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
    
    def get_size_mb(self) -> float:
        """Get model size in MB."""
        total_params = sum(p.numel() for p in self.parameters())
        return total_params * 4 / (1024 * 1024)


class ResourceClient:
    """Client with resource awareness."""
    
    def __init__(
        self,
        client_id: int,
        dataset: ResourceDataset,
        config: ResourceConfig,
        resources: ClientResources
    ):
        self.client_id = client_id
        self.dataset = dataset
        self.config = config
        self.resources = resources
    
    def can_train(self, model_size_mb: float) -> bool:
        """Check if client can train given model."""
        # Memory check
        if model_size_mb * 3 > self.resources.memory_mb:  # Need ~3x for training
            return False
        
        # Battery check
        if self.resources.battery_level < 0.1 and not self.resources.is_charging:
            return False
        
        return True
    
    def train(self, model: nn.Module) -> Dict[str, Any]:
        """Train with resource tracking."""
        start_time = time.time()
        
        # Adjust epochs based on compute speed
        epochs = max(1, int(self.config.local_epochs * self.resources.compute_speed))
        
        local = copy.deepcopy(model)
        optimizer = torch.optim.SGD(
            local.parameters(),
            lr=self.config.learning_rate
        )
        
        batch_size = min(
            self.config.batch_size,
            len(self.dataset)
        )
        
        loader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=True
        )
        
        local.train()
        total_loss = 0.0
        num_batches = 0
        
        for _ in range(epochs):
            for x, y in loader:
                optimizer.zero_grad()
                loss = F.cross_entropy(local(x), y)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
        
        training_time = time.time() - start_time
        
        # Simulate battery drain
        if not self.resources.is_charging:
            self.resources.battery_level -= 0.02 * epochs
            self.resources.battery_level = max(0, self.resources.battery_level)
        
        return {
            "state_dict": {k: v.cpu() for k, v in local.state_dict().items()},
            "num_samples": len(self.dataset),
            "avg_loss": total_loss / num_batches,
            "client_id": self.client_id,
            "training_time": training_time,
            "epochs_completed": epochs
        }


class ResourceServer:
    """Server with resource-aware client selection."""
    
    def __init__(
        self,
        model: nn.Module,
        clients: List[ResourceClient],
        test_data: ResourceDataset,
        config: ResourceConfig
    ):
        self.model = model
        self.clients = clients
        self.test_data = test_data
        self.config = config
        
        # Resource management
        self.monitor = ResourceMonitor(len(clients))
        for c in clients:
            self.monitor.register_profile(c.client_id, c.resources)
        
        self.scheduler = ResourceScheduler(self.monitor, config)
        self.history: List[Dict] = []
    
    def aggregate(self, updates: List[Dict]) -> None:
        """Aggregate updates."""
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
        loader = DataLoader(self.test_data, batch_size=64)
        
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in loader:
                pred = self.model(x).argmax(dim=1)
                correct += (pred == y).sum().item()
                total += len(y)
        
        return {"accuracy": correct / total}
    
    def train(self, strategy: str = "fastest") -> List[Dict]:
        """Run resource-aware training."""
        logger.info(f"Starting FL with {strategy} selection strategy")
        
        model_size = self.model.get_size_mb()
        
        for round_num in range(self.config.num_rounds):
            # Get available clients
            available = self.monitor.get_available_clients()
            available = [cid for cid in available if self.clients[cid].can_train(model_size)]
            
            if len(available) < 2:
                logger.warning("Not enough available clients")
                continue
            
            # Select clients
            selected_ids = self.scheduler.select_clients(
                available,
                self.config.clients_per_round,
                strategy=strategy
            )
            
            # Collect updates
            updates = []
            round_times = []
            for cid in selected_ids:
                result = self.clients[cid].train(self.model)
                updates.append(result)
                round_times.append(result["training_time"])
                
                # Record usage
                self.monitor.record_usage(cid, {
                    "time": result["training_time"],
                    "epochs": result["epochs_completed"]
                })
            
            # Aggregate
            self.aggregate(updates)
            
            # Evaluate
            metrics = self.evaluate()
            
            record = {
                "round": round_num,
                **metrics,
                "num_clients": len(updates),
                "max_time": max(round_times),
                "avg_time": np.mean(round_times)
            }
            self.history.append(record)
            
            if (round_num + 1) % 10 == 0:
                logger.info(
                    f"Round {round_num + 1}: acc={metrics['accuracy']:.4f}, "
                    f"time={record['max_time']:.2f}s"
                )
        
        return self.history


def main():
    """Main entry point."""
    print("=" * 60)
    print("Tutorial 107: FL Resource Management")
    print("=" * 60)
    
    config = ResourceConfig()
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    
    # Create clients with varied resources
    clients = []
    for i in range(config.num_clients):
        resources = ClientResources(
            compute_speed=np.random.uniform(0.5, 2.0),
            memory_mb=np.random.uniform(128, 512),
            bandwidth_mbps=np.random.uniform(10, 100),
            battery_level=np.random.uniform(0.3, 1.0),
            is_charging=np.random.choice([True, False], p=[0.3, 0.7])
        )
        
        dataset = ResourceDataset(client_id=i, dim=config.input_dim, seed=config.seed)
        client = ResourceClient(i, dataset, config, resources)
        clients.append(client)
    
    test_data = ResourceDataset(client_id=999, n=300, seed=999)
    model = ResourceModel(config)
    
    # Train with different strategies
    for strategy in ["fastest", "fair", "energy_aware"]:
        print(f"\n--- Strategy: {strategy} ---")
        
        # Reset clients
        for c in clients:
            c.resources.battery_level = np.random.uniform(0.3, 1.0)
        
        server = ResourceServer(model, clients, test_data, config)
        history = server.train(strategy=strategy)
        
        avg_time = np.mean([h["max_time"] for h in history])
        print(f"Final Accuracy: {history[-1]['accuracy']:.4f}")
        print(f"Avg Round Time: {avg_time:.2f}s")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
```

---

## Key Insights

### Resource Management Strategies

1. **Fastest**: Minimize round time
2. **Fair**: Equal participation
3. **Energy-aware**: Preserve battery
4. **Adaptive**: Combine based on conditions

### Best Practices

- Profile clients before training
- Adapt to resource changes
- Balance speed and fairness
- Consider energy constraints

---

## Exercises

1. **Exercise 1**: Add dynamic batch size
2. **Exercise 2**: Implement model compression
3. **Exercise 3**: Design priority scheduling
4. **Exercise 4**: Add thermal throttling

---

## References

1. Li, T., et al. (2020). Federated optimization. *MLSys*.
2. Nishio, T., & Yonetani, R. (2019). Client selection for FL. In *ICC*.
3. Yang, Y., et al. (2021). Resource-efficient FL. *IEEE TMC*.

---

*Copyright 2026 Olaf Yunus Laitinen Imanov and Contributors. Released under EUPL 1.2.*
