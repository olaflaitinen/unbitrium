# Tutorial 128: FL Deployment

---

## Metadata

| Property | Value |
|----------|-------|
| **Tutorial ID** | 128 |
| **Title** | Federated Learning Deployment |
| **Category** | Engineering |
| **Difficulty** | Advanced |
| **Duration** | 120 minutes |
| **Prerequisites** | Tutorial 001-127 |
| **Author** | Unbitrium Contributors |
| **Last Updated** | January 2026 |

---

## Learning Objectives

By completing this tutorial, you will be able to:

1. **Understand** FL deployment considerations
2. **Implement** production-ready FL systems
3. **Design** scalable FL infrastructure
4. **Manage** model lifecycle in FL
5. **Deploy** FL across diverse environments

---

## Prerequisites

Before starting this tutorial, ensure you have:

- Completed tutorials 001-127
- Understanding of FL fundamentals
- Knowledge of deployment practices
- Familiarity with production systems

---

## Background and Theory

### Deployment Considerations

```
FL Deployment:
├── Infrastructure
│   ├── Server provisioning
│   ├── Client SDK
│   └── Communication
├── Model Management
│   ├── Versioning
│   ├── Updates
│   └── Rollback
├── Security
│   ├── Authentication
│   ├── Encryption
│   └── Access control
└── Operations
    ├── Monitoring
    ├── Alerting
    └── Scaling
```

---

## Implementation Code

```python
#!/usr/bin/env python3
"""
Tutorial 128: Federated Learning Deployment

This module implements production-ready FL deployment
with versioning, rollback, and management.

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
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DeployConfig:
    """Deployment configuration."""
    
    num_rounds: int = 30
    num_clients: int = 15
    clients_per_round: int = 8
    
    input_dim: int = 32
    hidden_dim: int = 64
    num_classes: int = 10
    
    learning_rate: float = 0.01
    batch_size: int = 32
    local_epochs: int = 3
    
    # Deployment parameters
    min_accuracy: float = 0.7  # Minimum for deployment
    rollback_threshold: float = 0.1  # Performance drop threshold
    
    seed: int = 42


class ModelVersion:
    """Model version with metadata."""
    
    def __init__(
        self,
        version: str,
        model_state: Dict[str, torch.Tensor],
        metrics: Dict[str, float],
        timestamp: Optional[datetime] = None
    ):
        self.version = version
        self.model_state = model_state
        self.metrics = metrics
        self.timestamp = timestamp or datetime.utcnow()
        self.checksum = self._compute_checksum()
    
    def _compute_checksum(self) -> str:
        """Compute model checksum."""
        hasher = hashlib.sha256()
        for key in sorted(self.model_state.keys()):
            hasher.update(key.encode())
            hasher.update(self.model_state[key].numpy().tobytes())
        return hasher.hexdigest()[:16]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "checksum": self.checksum,
            "metrics": self.metrics,
            "timestamp": self.timestamp.isoformat()
        }


class ModelRegistry:
    """Registry for model versions."""
    
    def __init__(self):
        self.versions: Dict[str, ModelVersion] = {}
        self.current_version: Optional[str] = None
        self.history: List[str] = []
    
    def register(
        self,
        version: str,
        model_state: Dict[str, torch.Tensor],
        metrics: Dict[str, float]
    ) -> ModelVersion:
        """Register new model version."""
        mv = ModelVersion(version, model_state, metrics)
        self.versions[version] = mv
        self.history.append(version)
        logger.info(f"Registered model version: {version} (acc={metrics.get('accuracy', 0):.4f})")
        return mv
    
    def promote(self, version: str) -> bool:
        """Promote version to current."""
        if version not in self.versions:
            return False
        
        self.current_version = version
        logger.info(f"Promoted model version: {version}")
        return True
    
    def get_current(self) -> Optional[ModelVersion]:
        """Get current model version."""
        if self.current_version:
            return self.versions.get(self.current_version)
        return None
    
    def rollback(self, n: int = 1) -> bool:
        """Rollback to previous version."""
        if len(self.history) <= n:
            return False
        
        target = self.history[-(n + 1)]
        self.current_version = target
        logger.warning(f"Rolled back to version: {target}")
        return True
    
    def list_versions(self) -> List[Dict[str, Any]]:
        """List all versions."""
        return [v.to_dict() for v in self.versions.values()]


class DeploymentGate:
    """Gate model promotion based on criteria."""
    
    def __init__(self, config: DeployConfig):
        self.config = config
        self.baseline_accuracy: Optional[float] = None
    
    def evaluate(
        self,
        metrics: Dict[str, float],
        current_metrics: Optional[Dict[str, float]] = None
    ) -> Tuple[bool, str]:
        """Evaluate if model is ready for deployment."""
        accuracy = metrics.get("accuracy", 0)
        
        # Check minimum accuracy
        if accuracy < self.config.min_accuracy:
            return False, f"Below minimum ({accuracy:.4f} < {self.config.min_accuracy})"
        
        # Check for regression
        if current_metrics:
            current_acc = current_metrics.get("accuracy", 0)
            if accuracy < current_acc - self.config.rollback_threshold:
                return False, f"Regression detected ({accuracy:.4f} < {current_acc:.4f})"
        
        # Update baseline
        if self.baseline_accuracy is None:
            self.baseline_accuracy = accuracy
        
        return True, "All checks passed"


class DeployDataset(Dataset):
    """Dataset for deployment experiments."""
    
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


class DeployModel(nn.Module):
    """Deployable model."""
    
    def __init__(self, config: DeployConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DeployClient:
    """Client for deployment experiments."""
    
    def __init__(
        self,
        client_id: int,
        dataset: DeployDataset,
        config: DeployConfig
    ):
        self.client_id = client_id
        self.dataset = dataset
        self.config = config
    
    def train(self, model: nn.Module) -> Dict[str, Any]:
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
        
        return {
            "state_dict": {k: v.cpu() for k, v in local.state_dict().items()},
            "num_samples": len(self.dataset),
            "avg_loss": total_loss / num_batches
        }


class DeployServer:
    """Server with deployment management."""
    
    def __init__(
        self,
        model: nn.Module,
        clients: List[DeployClient],
        test_data: DeployDataset,
        config: DeployConfig
    ):
        self.model = model
        self.clients = clients
        self.test_data = test_data
        self.config = config
        
        self.registry = ModelRegistry()
        self.gate = DeploymentGate(config)
        self.history: List[Dict] = []
    
    def aggregate(self, updates: List[Dict]) -> None:
        total_samples = sum(u["num_samples"] for u in updates)
        new_state = {}
        
        for key in updates[0]["state_dict"]:
            new_state[key] = sum(
                (u["num_samples"] / total_samples) * u["state_dict"][key].float()
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
    
    def train_and_deploy(self) -> List[Dict]:
        """Run FL with deployment management."""
        logger.info(f"Starting FL deployment with {len(self.clients)} clients")
        
        for round_num in range(self.config.num_rounds):
            n = min(self.config.clients_per_round, len(self.clients))
            indices = np.random.choice(len(self.clients), n, replace=False)
            selected = [self.clients[i] for i in indices]
            
            updates = [c.train(self.model) for c in selected]
            self.aggregate(updates)
            
            metrics = self.evaluate()
            
            # Register new version
            version = f"v{round_num + 1}.0.0"
            model_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
            mv = self.registry.register(version, model_state, metrics)
            
            # Evaluate for promotion
            current_mv = self.registry.get_current()
            current_metrics = current_mv.metrics if current_mv else None
            
            should_promote, reason = self.gate.evaluate(metrics, current_metrics)
            
            if should_promote:
                self.registry.promote(version)
            else:
                logger.warning(f"Not promoting {version}: {reason}")
                if current_mv:
                    self.model.load_state_dict(current_mv.model_state)
            
            record = {
                "round": round_num,
                **metrics,
                "version": version,
                "promoted": should_promote
            }
            self.history.append(record)
            
            if (round_num + 1) % 10 == 0:
                logger.info(f"Round {round_num + 1}: acc={metrics['accuracy']:.4f}")
        
        return self.history


def main():
    """Main entry point."""
    print("=" * 60)
    print("Tutorial 128: FL Deployment")
    print("=" * 60)
    
    config = DeployConfig()
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    
    clients = []
    for i in range(config.num_clients):
        dataset = DeployDataset(client_id=i, dim=config.input_dim, seed=config.seed)
        client = DeployClient(i, dataset, config)
        clients.append(client)
    
    test_data = DeployDataset(client_id=999, n=300, seed=999)
    model = DeployModel(config)
    
    server = DeployServer(model, clients, test_data, config)
    history = server.train_and_deploy()
    
    print("\n" + "=" * 60)
    print("Deployment Summary")
    print(f"Total versions: {len(server.registry.versions)}")
    print(f"Current version: {server.registry.current_version}")
    print(f"Final accuracy: {history[-1]['accuracy']:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
```

---

## Key Insights

### Deployment Best Practices

1. **Version everything**: Track all models
2. **Gate promotion**: Quality checks
3. **Enable rollback**: Quick recovery
4. **Monitor continuously**: Detect issues

---

## Exercises

1. **Exercise 1**: Add A/B testing
2. **Exercise 2**: Implement canary deployment
3. **Exercise 3**: Design blue-green deployment
4. **Exercise 4**: Add feature flags

---

## References

1. Bonawitz, K., et al. (2019). Towards FL at scale. In *MLSys*.
2. Google AI (2022). FL production guide.
3. TensorFlow Federated documentation.

---

*Copyright 2026 Olaf Yunus Laitinen Imanov and Contributors. Released under EUPL 1.2.*
