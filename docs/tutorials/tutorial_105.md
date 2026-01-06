# Tutorial 105: FL Cross-Silo Systems

---

## Metadata

| Property | Value |
|----------|-------|
| **Tutorial ID** | 105 |
| **Title** | FL Cross-Silo Systems |
| **Category** | Systems |
| **Difficulty** | Advanced |
| **Duration** | 120 minutes |
| **Prerequisites** | Tutorial 001-104 |
| **Author** | Unbitrium Contributors |
| **Last Updated** | January 2026 |

---

## Learning Objectives

By completing this tutorial, you will be able to:

1. **Understand** cross-silo FL
2. **Implement** enterprise FL systems
3. **Design** multi-organization collaboration
4. **Handle** institutional constraints
5. **Deploy** production cross-silo FL

---

## Background and Theory

### Cross-Silo Characteristics

```
Cross-Silo FL:
├── Participants
│   ├── Organizations
│   ├── Data centers
│   └── Enterprises
├── Resources
│   ├── Reliable connectivity
│   ├── Large datasets
│   └── Dedicated compute
├── Requirements
│   ├── Data governance
│   ├── Contractual agreements
│   ├── Regulatory compliance
│   └── Audit trails
└── Coordination
    ├── Sync training
    ├── Full participation
    └── Known identities
```

---

## Implementation Code

```python
#!/usr/bin/env python3
"""
Tutorial 105: FL Cross-Silo Systems

This module implements cross-silo federated
learning for enterprise collaboration.

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
import copy
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SiloProfile:
    """Enterprise silo profile."""
    
    silo_id: str
    organization: str
    data_size: int
    compute_tier: str
    region: str = "default"
    contribution_weight: float = 1.0


@dataclass
class CrossSiloConfig:
    """Cross-silo configuration."""
    
    num_rounds: int = 50
    num_silos: int = 5
    
    input_dim: int = 64
    hidden_dim: int = 128
    num_classes: int = 10
    
    learning_rate: float = 0.01
    batch_size: int = 64
    local_epochs: int = 5
    
    # Enterprise features
    enable_audit: bool = True
    verify_contributions: bool = True
    
    seed: int = 42


class AuditLog:
    """Audit trail for FL operations."""
    
    def __init__(self):
        self.entries: List[Dict] = []
    
    def log(
        self,
        action: str,
        silo_id: str,
        round_num: int,
        details: Dict
    ) -> None:
        self.entries.append({
            "action": action,
            "silo_id": silo_id,
            "round": round_num,
            "details": details
        })
    
    def get_silo_history(self, silo_id: str) -> List[Dict]:
        return [e for e in self.entries if e["silo_id"] == silo_id]


class EnterpriseModel(nn.Module):
    """Enterprise-grade model."""
    
    def __init__(self, config: CrossSiloConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(config.hidden_dim // 2, config.num_classes)
        )
    
    def forward(self, x): return self.net(x)


class SiloDataset(Dataset):
    """Enterprise silo dataset."""
    
    def __init__(self, silo_id: str, n: int = 1000, dim: int = 64, classes: int = 10, seed: int = 0):
        np.random.seed(seed + hash(silo_id) % 1000)
        self.x = torch.randn(n, dim, dtype=torch.float32)
        self.y = torch.randint(0, classes, (n,), dtype=torch.long)
        for i in range(n):
            self.x[i, self.y[i].item() % dim] += 2.0
    
    def __len__(self): return len(self.y)
    def __getitem__(self, idx): return self.x[idx], self.y[idx]


class EnterpriseSilo:
    """Enterprise organization in cross-silo FL."""
    
    def __init__(self, profile: SiloProfile, config: CrossSiloConfig):
        self.profile = profile
        self.config = config
        
        data_size = profile.data_size
        self.dataset = SiloDataset(profile.silo_id, n=data_size, seed=config.seed)
    
    def train(self, model: nn.Module) -> Dict:
        """Train on silo data."""
        local = copy.deepcopy(model)
        optimizer = torch.optim.AdamW(local.parameters(), lr=self.config.learning_rate, weight_decay=0.01)
        loader = DataLoader(self.dataset, batch_size=self.config.batch_size, shuffle=True)
        
        local.train()
        total_loss = 0.0
        num_batches = 0
        
        for _ in range(self.config.local_epochs):
            for x, y in loader:
                optimizer.zero_grad()
                loss = F.cross_entropy(local(x), y)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(local.parameters(), 1.0)
                
                optimizer.step()
                total_loss += loss.item()
                num_batches += 1
        
        return {
            "state_dict": {k: v.cpu() for k, v in local.state_dict().items()},
            "num_samples": len(self.dataset),
            "avg_loss": total_loss / num_batches,
            "silo_id": self.profile.silo_id,
            "weight": self.profile.contribution_weight
        }
    
    def evaluate(self, model: nn.Module) -> Dict[str, float]:
        """Evaluate on local data."""
        model.eval()
        loader = DataLoader(self.dataset, batch_size=64)
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in loader:
                pred = model(x).argmax(dim=1)
                correct += (pred == y).sum().item()
                total += len(y)
        return {"local_accuracy": correct / total}


class CrossSiloServer:
    """Server for cross-silo FL."""
    
    def __init__(self, model: nn.Module, silos: List[EnterpriseSilo], test_data: SiloDataset, config: CrossSiloConfig):
        self.model = model
        self.silos = silos
        self.test_data = test_data
        self.config = config
        
        self.audit = AuditLog() if config.enable_audit else None
        self.history: List[Dict] = []
    
    def aggregate(self, updates: List[Dict]) -> None:
        total_weight = sum(u["num_samples"] * u["weight"] for u in updates)
        new_state = {}
        
        for key in updates[0]["state_dict"]:
            new_state[key] = sum(
                (u["num_samples"] * u["weight"] / total_weight) * u["state_dict"][key].float()
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
        logger.info(f"Starting cross-silo FL ({len(self.silos)} organizations)")
        
        for round_num in range(self.config.num_rounds):
            updates = []
            
            for silo in self.silos:
                update = silo.train(self.model)
                updates.append(update)
                
                if self.audit:
                    self.audit.log(
                        "TRAINING_COMPLETE",
                        silo.profile.silo_id,
                        round_num,
                        {"samples": update["num_samples"], "loss": update["avg_loss"]}
                    )
            
            self.aggregate(updates)
            
            metrics = self.evaluate()
            
            # Evaluate on each silo
            silo_metrics = {s.profile.silo_id: s.evaluate(self.model) for s in self.silos}
            
            record = {"round": round_num, **metrics, "silo_metrics": silo_metrics}
            self.history.append(record)
            
            if (round_num + 1) % 10 == 0:
                logger.info(f"Round {round_num + 1}: acc={metrics['accuracy']:.4f}")
        
        return self.history


def main():
    print("=" * 60)
    print("Tutorial 105: FL Cross-Silo Systems")
    print("=" * 60)
    
    config = CrossSiloConfig()
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    
    # Create enterprise silos
    profiles = [
        SiloProfile("hospital_a", "HealthCorp", 2000, "enterprise"),
        SiloProfile("hospital_b", "MedCenter", 1500, "standard"),
        SiloProfile("research_lab", "UniMed", 3000, "hpc"),
        SiloProfile("clinic_network", "ClinicGroup", 1000, "standard"),
        SiloProfile("pharma_co", "PharmaCorp", 2500, "enterprise"),
    ]
    
    silos = [EnterpriseSilo(p, config) for p in profiles]
    test_data = SiloDataset("test", n=500, seed=999)
    model = EnterpriseModel(config)
    
    server = CrossSiloServer(model, silos, test_data, config)
    history = server.train()
    
    print("\n" + "=" * 60)
    print("Cross-Silo Training Complete")
    print(f"Final accuracy: {history[-1]['accuracy']:.4f}")
    print(f"Audit entries: {len(server.audit.entries) if server.audit else 0}")
    print("=" * 60)


if __name__ == "__main__":
    main()
```

---

## Key Insights

### Cross-Silo Best Practices

1. **Full participation**: All silos train per round
2. **Contribution weighting**: Fair incentives
3. **Audit trails**: Regulatory compliance
4. **Data governance**: Respect agreements

---

## Exercises

1. **Exercise 1**: Add contribution verification
2. **Exercise 2**: Implement fair incentives
3. **Exercise 3**: Design governance dashboard
4. **Exercise 4**: Add multi-region support

---

## References

1. Kairouz, P., et al. (2021). Advances and open problems in FL. *Foundations and Trends in ML*.
2. Li, Q., et al. (2020). A survey on FL systems. In *TIST*.
3. FL for enterprise collaboration (Google AI Blog).

---

*Copyright 2026 Olaf Yunus Laitinen Imanov and Contributors. Released under EUPL 1.2.*
