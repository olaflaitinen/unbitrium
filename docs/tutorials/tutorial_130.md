# Tutorial 130: FL Version Control

---

## Metadata

| Property | Value |
|----------|-------|
| **Tutorial ID** | 130 |
| **Title** | Federated Learning Version Control |
| **Category** | Engineering |
| **Difficulty** | Advanced |
| **Duration** | 90 minutes |
| **Prerequisites** | Tutorial 001-129 |
| **Author** | Unbitrium Contributors |
| **Last Updated** | January 2026 |

---

## Learning Objectives

By completing this tutorial, you will be able to:

1. **Understand** versioning in FL
2. **Implement** model versioning systems
3. **Design** reproducible FL experiments
4. **Track** model lineage
5. **Deploy** versioned FL pipelines

---

## Prerequisites

Before starting this tutorial, ensure you have:

- Completed tutorials 001-129
- Understanding of FL fundamentals
- Knowledge of version control
- Familiarity with MLOps

---

## Background and Theory

### What to Version in FL

```
FL Versioning:
├── Models
│   ├── Global model versions
│   ├── Client model snapshots
│   └── Aggregated checkpoints
├── Configuration
│   ├── Hyperparameters
│   ├── Client selection
│   └── Aggregation parameters
├── Data
│   ├── Data schemas
│   ├── Preprocessing
│   └── Feature versions
└── Experiments
    ├── Run configurations
    ├── Metrics history
    └── Artifacts
```

---

## Implementation Code

```python
#!/usr/bin/env python3
"""
Tutorial 130: FL Version Control

This module implements version control for FL
including model versioning and experiment tracking.

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
class VCConfig:
    """Version control configuration."""
    
    num_rounds: int = 30
    num_clients: int = 10
    clients_per_round: int = 5
    
    input_dim: int = 32
    hidden_dim: int = 64
    num_classes: int = 10
    
    learning_rate: float = 0.01
    batch_size: int = 32
    local_epochs: int = 3
    
    seed: int = 42


@dataclass
class ModelCheckpoint:
    """Model checkpoint with metadata."""
    
    version: str
    round_num: int
    state_dict: Dict[str, torch.Tensor]
    metrics: Dict[str, float]
    config: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.utcnow)
    parent_version: Optional[str] = None
    
    def get_hash(self) -> str:
        """Compute checkpoint hash."""
        hasher = hashlib.sha256()
        for key in sorted(self.state_dict.keys()):
            hasher.update(key.encode())
            hasher.update(self.state_dict[key].numpy().tobytes())
        return hasher.hexdigest()[:12]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "round": self.round_num,
            "hash": self.get_hash(),
            "metrics": self.metrics,
            "timestamp": self.timestamp.isoformat(),
            "parent": self.parent_version
        }


class VersionStore:
    """Store and manage versions."""
    
    def __init__(self):
        self.checkpoints: Dict[str, ModelCheckpoint] = {}
        self.tags: Dict[str, str] = {}  # Tag -> version
        self.head: Optional[str] = None
    
    def save(
        self,
        version: str,
        round_num: int,
        state_dict: Dict[str, torch.Tensor],
        metrics: Dict[str, float],
        config: Dict[str, Any]
    ) -> ModelCheckpoint:
        """Save new checkpoint."""
        parent = self.head
        
        checkpoint = ModelCheckpoint(
            version=version,
            round_num=round_num,
            state_dict={k: v.cpu().clone() for k, v in state_dict.items()},
            metrics=metrics,
            config=config,
            parent_version=parent
        )
        
        self.checkpoints[version] = checkpoint
        self.head = version
        
        return checkpoint
    
    def load(self, version: str) -> Optional[ModelCheckpoint]:
        """Load checkpoint by version."""
        return self.checkpoints.get(version)
    
    def tag(self, tag: str, version: Optional[str] = None) -> None:
        """Tag a version."""
        version = version or self.head
        if version:
            self.tags[tag] = version
            logger.info(f"Tagged {version} as '{tag}'")
    
    def get_by_tag(self, tag: str) -> Optional[ModelCheckpoint]:
        """Get checkpoint by tag."""
        version = self.tags.get(tag)
        return self.checkpoints.get(version) if version else None
    
    def list_versions(self) -> List[Dict[str, Any]]:
        """List all versions."""
        return [cp.to_dict() for cp in self.checkpoints.values()]
    
    def get_lineage(self, version: str) -> List[str]:
        """Get version lineage."""
        lineage = []
        current = version
        
        while current:
            lineage.append(current)
            cp = self.checkpoints.get(current)
            current = cp.parent_version if cp else None
        
        return lineage


class ExperimentTracker:
    """Track FL experiments."""
    
    def __init__(self, experiment_name: str):
        self.name = experiment_name
        self.runs: List[Dict[str, Any]] = []
        self.current_run: Optional[Dict[str, Any]] = None
    
    def start_run(self, config: Dict[str, Any]) -> str:
        """Start new experiment run."""
        run_id = f"run_{len(self.runs) + 1}"
        
        self.current_run = {
            "id": run_id,
            "config": config,
            "start_time": datetime.utcnow().isoformat(),
            "metrics": [],
            "status": "running"
        }
        
        logger.info(f"Started run: {run_id}")
        return run_id
    
    def log_metrics(self, round_num: int, metrics: Dict[str, float]) -> None:
        """Log round metrics."""
        if self.current_run:
            self.current_run["metrics"].append({
                "round": round_num,
                **metrics
            })
    
    def end_run(self, status: str = "completed") -> None:
        """End current run."""
        if self.current_run:
            self.current_run["status"] = status
            self.current_run["end_time"] = datetime.utcnow().isoformat()
            self.runs.append(self.current_run)
            self.current_run = None
    
    def compare_runs(
        self,
        run_ids: List[str],
        metric: str = "accuracy"
    ) -> Dict[str, List[float]]:
        """Compare runs by metric."""
        comparison = {}
        
        for run in self.runs:
            if run["id"] in run_ids:
                values = [m.get(metric, 0) for m in run["metrics"]]
                comparison[run["id"]] = values
        
        return comparison


class VCDataset(Dataset):
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


class VCModel(nn.Module):
    def __init__(self, config: VCConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class VCClient:
    def __init__(
        self,
        client_id: int,
        dataset: VCDataset,
        config: VCConfig
    ):
        self.client_id = client_id
        self.dataset = dataset
        self.config = config
    
    def train(self, model: nn.Module) -> Dict[str, Any]:
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
        
        return {
            "state_dict": {k: v.cpu() for k, v in local.state_dict().items()},
            "num_samples": len(self.dataset),
            "avg_loss": total_loss / num_batches
        }


class VCServer:
    def __init__(
        self,
        model: nn.Module,
        clients: List[VCClient],
        test_data: VCDataset,
        config: VCConfig
    ):
        self.model = model
        self.clients = clients
        self.test_data = test_data
        self.config = config
        
        self.version_store = VersionStore()
        self.tracker = ExperimentTracker("fl_experiment")
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
    
    def train(self) -> List[Dict]:
        config_dict = {
            "num_rounds": self.config.num_rounds,
            "clients_per_round": self.config.clients_per_round,
            "learning_rate": self.config.learning_rate
        }
        
        self.tracker.start_run(config_dict)
        
        for round_num in range(self.config.num_rounds):
            n = min(self.config.clients_per_round, len(self.clients))
            indices = np.random.choice(len(self.clients), n, replace=False)
            selected = [self.clients[i] for i in indices]
            
            updates = [c.train(self.model) for c in selected]
            self.aggregate(updates)
            
            metrics = self.evaluate()
            
            # Save checkpoint
            version = f"v{round_num + 1}"
            self.version_store.save(
                version, round_num,
                self.model.state_dict(), metrics, config_dict
            )
            
            self.tracker.log_metrics(round_num, metrics)
            
            record = {"round": round_num, **metrics, "version": version}
            self.history.append(record)
            
            if (round_num + 1) % 10 == 0:
                logger.info(f"Round {round_num + 1}: acc={metrics['accuracy']:.4f}")
        
        # Tag best version
        best_round = max(self.history, key=lambda x: x["accuracy"])
        self.version_store.tag("best", best_round["version"])
        
        self.tracker.end_run()
        return self.history


def main():
    print("=" * 60)
    print("Tutorial 130: FL Version Control")
    print("=" * 60)
    
    config = VCConfig()
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    
    clients = []
    for i in range(config.num_clients):
        dataset = VCDataset(client_id=i, dim=config.input_dim, seed=config.seed)
        client = VCClient(i, dataset, config)
        clients.append(client)
    
    test_data = VCDataset(client_id=999, n=300, seed=999)
    model = VCModel(config)
    
    server = VCServer(model, clients, test_data, config)
    history = server.train()
    
    print("\n" + "=" * 60)
    print("Version Control Summary")
    print(f"Total versions: {len(server.version_store.checkpoints)}")
    print(f"Current HEAD: {server.version_store.head}")
    print(f"Tags: {server.version_store.tags}")
    
    best = server.version_store.get_by_tag("best")
    if best:
        print(f"Best version: {best.version} (acc={best.metrics['accuracy']:.4f})")
    print("=" * 60)


if __name__ == "__main__":
    main()
```

---

## Key Insights

### Version Control Best Practices

1. **Version everything**: Models, configs, metrics
2. **Tag important versions**: best, production
3. **Track lineage**: Parent-child relationships
4. **Enable reproducibility**: Seed everything

---

## Exercises

1. **Exercise 1**: Implement branching
2. **Exercise 2**: Add diff between versions
3. **Exercise 3**: Design rollback functionality
4. **Exercise 4**: Add version comparison

---

## References

1. MLflow documentation
2. DVC for model versioning
3. Weights & Biases experiment tracking

---

*Copyright 2026 Olaf Yunus Laitinen Imanov and Contributors. Released under EUPL 1.2.*
