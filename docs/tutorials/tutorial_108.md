# Tutorial 108: FL Fault Tolerance

---

## Metadata

| Property | Value |
|----------|-------|
| **Tutorial ID** | 108 |
| **Title** | Federated Learning Fault Tolerance |
| **Category** | Systems |
| **Difficulty** | Advanced |
| **Duration** | 120 minutes |
| **Prerequisites** | Tutorial 001-107 |
| **Author** | Unbitrium Contributors |
| **Last Updated** | January 2026 |

---

## Learning Objectives

By completing this tutorial, you will be able to:

1. **Understand** fault tolerance in FL systems
2. **Implement** failure detection mechanisms
3. **Design** recovery strategies
4. **Handle** client dropouts gracefully
5. **Deploy** resilient FL systems

---

## Prerequisites

Before starting this tutorial, ensure you have:

- Completed tutorials 001-107
- Understanding of FL fundamentals
- Knowledge of distributed systems
- Familiarity with failure modes

---

## Background and Theory

### Failure Types in FL

```
FL Failure Types:
├── Client Failures
│   ├── Crash (device offline)
│   ├── Timeout (slow response)
│   ├── Straggler (delayed updates)
│   └── Drop (network issues)
├── Server Failures
│   ├── Coordinator crash
│   ├── Memory exhaustion
│   └── Network partition
├── Data Failures
│   ├── Corrupted updates
│   ├── Malformed messages
│   └── Version mismatch
└── Byzantine Failures
    ├── Malicious clients
    ├── Data poisoning
    └── Model poisoning
```

### Failure Handling Strategies

| Strategy | Failure Type | Approach |
|----------|--------------|----------|
| Timeout | Straggler | Wait limit |
| Retry | Transient | Exponential backoff |
| Replacement | Dropout | Select new client |
| Checkpointing | Any | Save progress |
| Replication | Critical | Redundancy |

---

## Implementation Code

```python
#!/usr/bin/env python3
"""
Tutorial 108: Federated Learning Fault Tolerance

This module implements fault-tolerant FL with failure
detection, recovery, and resilient aggregation.

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
import time
import random

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FailureType(Enum):
    """Types of client failures."""
    NONE = "none"
    CRASH = "crash"
    TIMEOUT = "timeout"
    CORRUPT = "corrupt"


@dataclass
class FaultConfig:
    """Configuration for fault-tolerant FL."""
    
    num_rounds: int = 50
    num_clients: int = 20
    clients_per_round: int = 10
    min_clients_for_aggregation: int = 5
    
    input_dim: int = 32
    hidden_dim: int = 64
    num_classes: int = 10
    
    learning_rate: float = 0.01
    batch_size: int = 32
    local_epochs: int = 3
    
    # Fault tolerance parameters
    max_retries: int = 3
    timeout_seconds: float = 10.0
    checkpoint_interval: int = 10
    
    # Failure simulation
    failure_rate: float = 0.2
    
    seed: int = 42


class FaultDataset(Dataset):
    """Dataset for fault tolerance experiments."""
    
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


class FaultModel(nn.Module):
    """Model for fault tolerance experiments."""
    
    def __init__(self, config: FaultConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Checkpoint:
    """Checkpoint manager for FL state."""
    
    def __init__(self, save_path: Optional[str] = None):
        self.save_path = save_path
        self.checkpoints: List[Dict] = []
    
    def save(
        self,
        model_state: Dict[str, torch.Tensor],
        round_num: int,
        metrics: Dict[str, float]
    ) -> None:
        """Save checkpoint."""
        checkpoint = {
            "model_state": {k: v.clone() for k, v in model_state.items()},
            "round": round_num,
            "metrics": metrics.copy(),
            "timestamp": time.time()
        }
        self.checkpoints.append(checkpoint)
        
        # Keep only last 3 checkpoints
        if len(self.checkpoints) > 3:
            self.checkpoints = self.checkpoints[-3:]
        
        logger.debug(f"Saved checkpoint at round {round_num}")
    
    def restore(self) -> Optional[Tuple[Dict, int]]:
        """Restore from latest checkpoint."""
        if not self.checkpoints:
            return None
        
        latest = self.checkpoints[-1]
        logger.info(f"Restoring from round {latest['round']}")
        return latest["model_state"], latest["round"]


class HealthMonitor:
    """Monitor client and system health."""
    
    def __init__(self, num_clients: int):
        self.num_clients = num_clients
        self.client_status: Dict[int, Dict] = {}
        self.failure_counts: Dict[int, int] = {i: 0 for i in range(num_clients)}
    
    def record_success(self, client_id: int, latency: float) -> None:
        """Record successful client response."""
        self.client_status[client_id] = {
            "status": "healthy",
            "last_seen": time.time(),
            "latency": latency
        }
        # Decay failure count
        self.failure_counts[client_id] = max(0, self.failure_counts[client_id] - 1)
    
    def record_failure(self, client_id: int, failure_type: FailureType) -> None:
        """Record client failure."""
        self.client_status[client_id] = {
            "status": "failed",
            "failure_type": failure_type.value,
            "timestamp": time.time()
        }
        self.failure_counts[client_id] += 1
    
    def is_reliable(self, client_id: int, threshold: int = 3) -> bool:
        """Check if client is considered reliable."""
        return self.failure_counts.get(client_id, 0) < threshold
    
    def get_healthy_clients(self) -> List[int]:
        """Get list of healthy clients."""
        return [
            cid for cid in range(self.num_clients)
            if self.is_reliable(cid)
        ]


class FaultClient:
    """Client with simulated failures."""
    
    def __init__(
        self,
        client_id: int,
        dataset: FaultDataset,
        config: FaultConfig,
        failure_rate: float = 0.2
    ):
        self.client_id = client_id
        self.dataset = dataset
        self.config = config
        self.failure_rate = failure_rate
        
        # Track state for retry logic
        self.last_update: Optional[Dict] = None
    
    def _simulate_failure(self) -> FailureType:
        """Simulate random failures."""
        if random.random() < self.failure_rate:
            failure_types = [FailureType.CRASH, FailureType.TIMEOUT, FailureType.CORRUPT]
            return random.choice(failure_types)
        return FailureType.NONE
    
    def train(self, model: nn.Module) -> Dict[str, Any]:
        """Train with possible failures."""
        # Simulate failure
        failure = self._simulate_failure()
        
        if failure == FailureType.CRASH:
            raise RuntimeError(f"Client {self.client_id} crashed!")
        
        if failure == FailureType.TIMEOUT:
            # Simulate timeout by returning None
            return {
                "status": "timeout",
                "client_id": self.client_id
            }
        
        # Actual training
        start_time = time.time()
        
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
        
        update = {
            "state_dict": {k: v.cpu() for k, v in local.state_dict().items()},
            "num_samples": len(self.dataset),
            "avg_loss": total_loss / num_batches,
            "client_id": self.client_id,
            "latency": time.time() - start_time,
            "status": "success"
        }
        
        # Simulate corrupt update
        if failure == FailureType.CORRUPT:
            for key in update["state_dict"]:
                update["state_dict"][key] = torch.randn_like(update["state_dict"][key])
            update["status"] = "corrupt"
        
        self.last_update = update
        return update


class FaultTolerantServer:
    """Server with fault tolerance."""
    
    def __init__(
        self,
        model: nn.Module,
        clients: List[FaultClient],
        test_data: FaultDataset,
        config: FaultConfig
    ):
        self.model = model
        self.clients = clients
        self.test_data = test_data
        self.config = config
        
        self.checkpoint = Checkpoint()
        self.monitor = HealthMonitor(len(clients))
        self.history: List[Dict] = []
    
    def _collect_with_retry(
        self,
        client: FaultClient,
        retries: int = 3
    ) -> Optional[Dict]:
        """Collect update with retry logic."""
        for attempt in range(retries):
            try:
                result = client.train(self.model)
                
                if result.get("status") == "success":
                    self.monitor.record_success(client.client_id, result.get("latency", 0))
                    return result
                
                elif result.get("status") == "timeout":
                    logger.warning(f"Client {client.client_id} timeout, attempt {attempt + 1}")
                    self.monitor.record_failure(client.client_id, FailureType.TIMEOUT)
                
                elif result.get("status") == "corrupt":
                    logger.warning(f"Client {client.client_id} corrupt update")
                    self.monitor.record_failure(client.client_id, FailureType.CORRUPT)
                    return None  # Don't retry corrupt
                
            except RuntimeError as e:
                logger.warning(f"Client {client.client_id} crashed: {e}")
                self.monitor.record_failure(client.client_id, FailureType.CRASH)
            
            # Exponential backoff
            time.sleep(0.01 * (2 ** attempt))
        
        return None
    
    def _validate_update(self, update: Dict) -> bool:
        """Validate client update."""
        if update.get("status") != "success":
            return False
        
        state_dict = update.get("state_dict", {})
        
        # Check for NaN or Inf
        for key, value in state_dict.items():
            if torch.isnan(value).any() or torch.isinf(value).any():
                logger.warning(f"Invalid values in update from {update['client_id']}")
                return False
        
        # Check for unusual magnitude
        for key, value in state_dict.items():
            if value.abs().max() > 1e6:
                logger.warning(f"Large values in update from {update['client_id']}")
                return False
        
        return True
    
    def aggregate(self, updates: List[Dict]) -> bool:
        """Aggregate valid updates."""
        valid_updates = [u for u in updates if self._validate_update(u)]
        
        if len(valid_updates) < self.config.min_clients_for_aggregation:
            logger.error(
                f"Not enough valid updates: {len(valid_updates)} < "
                f"{self.config.min_clients_for_aggregation}"
            )
            return False
        
        total_samples = sum(u["num_samples"] for u in valid_updates)
        new_state = {}
        
        for key in valid_updates[0]["state_dict"]:
            new_state[key] = sum(
                (u["num_samples"] / total_samples) * u["state_dict"][key].float()
                for u in valid_updates
            )
        
        self.model.load_state_dict(new_state)
        return True
    
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
    
    def train(self) -> List[Dict]:
        """Run fault-tolerant training."""
        logger.info(f"Starting fault-tolerant FL with {len(self.clients)} clients")
        
        for round_num in range(self.config.num_rounds):
            # Select reliable clients
            healthy = self.monitor.get_healthy_clients()
            n = min(self.config.clients_per_round, len(healthy))
            
            if n < self.config.min_clients_for_aggregation:
                logger.warning(f"Not enough healthy clients: {len(healthy)}")
                # Use all clients as fallback
                healthy = list(range(len(self.clients)))
                n = min(self.config.clients_per_round, len(healthy))
            
            selected = random.sample(healthy, n)
            
            # Collect updates with retry
            updates = []
            for cid in selected:
                result = self._collect_with_retry(self.clients[cid])
                if result:
                    updates.append(result)
            
            # Aggregate
            success = self.aggregate(updates)
            
            if not success:
                # Try to restore from checkpoint
                restored = self.checkpoint.restore()
                if restored:
                    state, prev_round = restored
                    self.model.load_state_dict(state)
                    logger.info(f"Restored to round {prev_round}")
                continue
            
            # Evaluate
            metrics = self.evaluate()
            
            # Checkpoint
            if (round_num + 1) % self.config.checkpoint_interval == 0:
                self.checkpoint.save(
                    self.model.state_dict(),
                    round_num,
                    metrics
                )
            
            record = {
                "round": round_num,
                **metrics,
                "successful_clients": len(updates),
                "selected_clients": n
            }
            self.history.append(record)
            
            if (round_num + 1) % 10 == 0:
                logger.info(
                    f"Round {round_num + 1}: "
                    f"acc={metrics['accuracy']:.4f}, "
                    f"success={len(updates)}/{n}"
                )
        
        return self.history


def main():
    """Main entry point."""
    print("=" * 60)
    print("Tutorial 108: FL Fault Tolerance")
    print("=" * 60)
    
    config = FaultConfig()
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)
    
    # Create clients with varying failure rates
    clients = []
    for i in range(config.num_clients):
        failure_rate = config.failure_rate if i < config.num_clients // 2 else config.failure_rate * 0.5
        
        dataset = FaultDataset(client_id=i, dim=config.input_dim, seed=config.seed)
        client = FaultClient(i, dataset, config, failure_rate)
        clients.append(client)
    
    # Test data
    test_data = FaultDataset(client_id=999, n=300, seed=999)
    
    # Model
    model = FaultModel(config)
    
    # Train
    server = FaultTolerantServer(model, clients, test_data, config)
    history = server.train()
    
    # Summary
    successful_rounds = len([h for h in history if "accuracy" in h])
    
    print("\n" + "=" * 60)
    print("Training Complete")
    print(f"Successful rounds: {successful_rounds}/{config.num_rounds}")
    print(f"Final Accuracy: {history[-1]['accuracy']:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
```

---

## Key Insights

### Fault Tolerance Best Practices

1. **Retry with backoff**: Handle transient failures
2. **Health tracking**: Avoid problematic clients
3. **Checkpointing**: Enable recovery
4. **Validation**: Detect corrupt updates
5. **Graceful degradation**: Work with partial updates

---

## Exercises

1. **Exercise 1**: Implement leader election
2. **Exercise 2**: Add Byzantine fault tolerance
3. **Exercise 3**: Design failover mechanisms
4. **Exercise 4**: Add distributed checkpointing

---

## References

1. Li, T., et al. (2020). Federated optimization. *MLSys*.
2. Bonawitz, K., et al. (2019). Towards FL at scale. In *MLSys*.
3. Velichko, A., et al. (2022). Fault-tolerant FL. *IEEE TPDS*.

---

*Copyright 2026 Olaf Yunus Laitinen Imanov and Contributors. Released under EUPL 1.2.*
