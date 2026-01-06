# Tutorial 110: FL Model Serving

---

## Metadata

| Property | Value |
|----------|-------|
| **Tutorial ID** | 110 |
| **Title** | Federated Learning Model Serving |
| **Category** | Systems |
| **Difficulty** | Advanced |
| **Duration** | 90 minutes |
| **Prerequisites** | Tutorial 001-109 |
| **Author** | Unbitrium Contributors |
| **Last Updated** | January 2026 |

---

## Learning Objectives

By completing this tutorial, you will be able to:

1. **Understand** model serving in FL contexts
2. **Implement** efficient model distribution
3. **Design** update delivery systems
4. **Analyze** serving latency
5. **Deploy** scalable FL serving

---

## Prerequisites

Before starting this tutorial, ensure you have:

- Completed tutorials 001-109
- Understanding of FL fundamentals
- Knowledge of model serving
- Familiarity with APIs

---

## Background and Theory

### FL Model Serving Challenges

```
FL Model Serving:
├── Model Distribution
│   ├── Push vs pull
│   ├── Incremental updates
│   └── Version management
├── Client Management
│   ├── Registration
│   ├── Capability discovery
│   └── Health monitoring
├── Update Delivery
│   ├── Full model sync
│   ├── Delta updates
│   └── Compressed delivery
└── Inference
    ├── Local inference
    ├── Server-assisted
    └── Hybrid
```

---

## Implementation Code

```python
#!/usr/bin/env python3
"""
Tutorial 110: FL Model Serving

This module implements model serving for FL including
model distribution and update management.

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
from datetime import datetime
import copy
import logging
import hashlib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ServingConfig:
    """Serving configuration."""
    
    num_rounds: int = 30
    num_clients: int = 15
    clients_per_round: int = 8
    
    input_dim: int = 32
    hidden_dim: int = 64
    num_classes: int = 10
    
    learning_rate: float = 0.01
    batch_size: int = 32
    local_epochs: int = 3
    
    seed: int = 42


class ModelVersion:
    """Model version for serving."""
    
    def __init__(
        self,
        version: str,
        state_dict: Dict[str, torch.Tensor],
        timestamp: Optional[datetime] = None
    ):
        self.version = version
        self.state_dict = {k: v.cpu().clone() for k, v in state_dict.items()}
        self.timestamp = timestamp or datetime.utcnow()
        self.checksum = self._compute_checksum()
    
    def _compute_checksum(self) -> str:
        hasher = hashlib.sha256()
        for key in sorted(self.state_dict.keys()):
            hasher.update(key.encode())
            hasher.update(self.state_dict[key].numpy().tobytes())
        return hasher.hexdigest()[:16]
    
    def compute_delta(self, other: 'ModelVersion') -> Dict[str, torch.Tensor]:
        """Compute delta to another version."""
        delta = {}
        for key in self.state_dict:
            if key in other.state_dict:
                delta[key] = self.state_dict[key] - other.state_dict[key]
            else:
                delta[key] = self.state_dict[key]
        return delta


class ModelServer:
    """Server for model distribution."""
    
    def __init__(self):
        self.versions: Dict[str, ModelVersion] = {}
        self.current_version: Optional[str] = None
        self.client_versions: Dict[int, str] = {}
    
    def publish(
        self,
        version: str,
        state_dict: Dict[str, torch.Tensor]
    ) -> ModelVersion:
        """Publish new model version."""
        mv = ModelVersion(version, state_dict)
        self.versions[version] = mv
        self.current_version = version
        logger.info(f"Published model version: {version}")
        return mv
    
    def get_current(self) -> Optional[ModelVersion]:
        """Get current version."""
        if self.current_version:
            return self.versions.get(self.current_version)
        return None
    
    def get_update(
        self,
        client_id: int,
        client_version: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get update for client."""
        current = self.get_current()
        if not current:
            return {"error": "No model available"}
        
        if client_version and client_version in self.versions:
            old = self.versions[client_version]
            delta = current.compute_delta(old)
            
            return {
                "type": "delta",
                "from_version": client_version,
                "to_version": current.version,
                "delta": delta,
                "checksum": current.checksum
            }
        
        return {
            "type": "full",
            "version": current.version,
            "state_dict": current.state_dict,
            "checksum": current.checksum
        }
    
    def register_client(self, client_id: int, version: str) -> None:
        """Register client version."""
        self.client_versions[client_id] = version
    
    def get_client_stats(self) -> Dict[str, int]:
        """Get client version statistics."""
        stats = {}
        for version in self.client_versions.values():
            stats[version] = stats.get(version, 0) + 1
        return stats


class ServingDataset(Dataset):
    def __init__(self, n: int = 200, dim: int = 32, classes: int = 10, seed: int = 0):
        np.random.seed(seed)
        self.x = torch.randn(n, dim, dtype=torch.float32)
        self.y = torch.randint(0, classes, (n,), dtype=torch.long)
        for i in range(n):
            self.x[i, self.y[i].item() % dim] += 2.0
    
    def __len__(self): return len(self.y)
    def __getitem__(self, idx): return self.x[idx], self.y[idx]


class ServingModel(nn.Module):
    def __init__(self, config: ServingConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.num_classes)
        )
    
    def forward(self, x): return self.net(x)


class ServingClient:
    """Client with model serving."""
    
    def __init__(
        self,
        client_id: int,
        dataset: ServingDataset,
        config: ServingConfig
    ):
        self.client_id = client_id
        self.dataset = dataset
        self.config = config
        
        self.current_version: Optional[str] = None
        self.local_model: Optional[nn.Module] = None
    
    def fetch_model(self, server: ModelServer) -> None:
        """Fetch model from server."""
        update = server.get_update(self.client_id, self.current_version)
        
        if "error" in update:
            logger.error(f"Client {self.client_id}: {update['error']}")
            return
        
        if update["type"] == "full":
            self.local_model = ServingModel(self.config)
            self.local_model.load_state_dict(update["state_dict"])
            self.current_version = update["version"]
        else:
            # Apply delta
            if self.local_model is None:
                logger.error("Cannot apply delta without base model")
                return
            
            state = self.local_model.state_dict()
            for key, delta in update["delta"].items():
                state[key] = state[key] + delta
            self.local_model.load_state_dict(state)
            self.current_version = update["to_version"]
        
        server.register_client(self.client_id, self.current_version)
    
    def train(self) -> Dict[str, Any]:
        """Train local model."""
        if not self.local_model:
            return {"error": "No local model"}
        
        local = copy.deepcopy(self.local_model)
        optimizer = torch.optim.SGD(local.parameters(), lr=self.config.learning_rate)
        loader = DataLoader(self.dataset, batch_size=self.config.batch_size, shuffle=True)
        
        local.train()
        total_loss, num_batches = 0.0, 0
        
        for _ in range(self.config.local_epochs):
            for x, y in loader:
                optimizer.zero_grad()
                loss = F.cross_entropy(local(x), y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                num_batches += 1
        
        return {
            "state_dict": {k: v.cpu() for k, v in local.state_dict().items()},
            "num_samples": len(self.dataset),
            "avg_loss": total_loss / num_batches,
            "client_id": self.client_id
        }
    
    def inference(self, x: torch.Tensor) -> torch.Tensor:
        """Local inference."""
        if not self.local_model:
            raise RuntimeError("No model loaded")
        
        self.local_model.eval()
        with torch.no_grad():
            return self.local_model(x)


class ServingCoordinator:
    """Coordinate FL with model serving."""
    
    def __init__(
        self,
        clients: List[ServingClient],
        test_data: ServingDataset,
        config: ServingConfig
    ):
        self.clients = clients
        self.test_data = test_data
        self.config = config
        
        self.model = ServingModel(config)
        self.model_server = ModelServer()
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
        logger.info(f"Starting FL with serving ({len(self.clients)} clients)")
        
        # Publish initial model
        self.model_server.publish("v0", self.model.state_dict())
        
        for round_num in range(self.config.num_rounds):
            version = f"v{round_num + 1}"
            
            n = min(self.config.clients_per_round, len(self.clients))
            indices = np.random.choice(len(self.clients), n, replace=False)
            selected = [self.clients[i] for i in indices]
            
            # Clients fetch model
            for client in selected:
                client.fetch_model(self.model_server)
            
            # Local training
            updates = [c.train() for c in selected]
            updates = [u for u in updates if "error" not in u]
            
            if updates:
                self.aggregate(updates)
                self.model_server.publish(version, self.model.state_dict())
            
            metrics = self.evaluate()
            
            record = {
                "round": round_num,
                **metrics,
                "version": version,
                "client_stats": self.model_server.get_client_stats()
            }
            self.history.append(record)
            
            if (round_num + 1) % 10 == 0:
                logger.info(f"Round {round_num + 1}: acc={metrics['accuracy']:.4f}")
        
        return self.history


def main():
    print("=" * 60)
    print("Tutorial 110: FL Model Serving")
    print("=" * 60)
    
    config = ServingConfig()
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    
    clients = [
        ServingClient(i, ServingDataset(seed=config.seed + i), config)
        for i in range(config.num_clients)
    ]
    test_data = ServingDataset(seed=999)
    
    coordinator = ServingCoordinator(clients, test_data, config)
    history = coordinator.train()
    
    print("\n" + "=" * 60)
    print("Serving Summary")
    print(f"Total versions: {len(coordinator.model_server.versions)}")
    print(f"Final accuracy: {history[-1]['accuracy']:.4f}")
    print(f"Client distribution: {history[-1]['client_stats']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
```

---

## Key Insights

### Model Serving Best Practices

1. **Delta updates**: Reduce bandwidth
2. **Version tracking**: Know client states
3. **Checksum validation**: Ensure integrity
4. **Graceful fallback**: Handle missing versions

---

## Exercises

1. **Exercise 1**: Add model compression
2. **Exercise 2**: Implement caching
3. **Exercise 3**: Add rollout strategies
4. **Exercise 4**: Design health checks

---

## References

1. TensorFlow Serving documentation
2. TorchServe documentation
3. Bonawitz, K., et al. (2019). Towards FL at scale. In *MLSys*.

---

*Copyright 2026 Olaf Yunus Laitinen Imanov and Contributors. Released under EUPL 1.2.*
