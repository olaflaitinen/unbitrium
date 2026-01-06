# Tutorial 142: FL Model Governance

---

## Metadata

| Property | Value |
|----------|-------|
| **Tutorial ID** | 142 |
| **Title** | Federated Learning Model Governance |
| **Category** | Governance |
| **Difficulty** | Advanced |
| **Duration** | 90 minutes |
| **Prerequisites** | Tutorial 001-141 |
| **Author** | Unbitrium Contributors |
| **Last Updated** | January 2026 |

---

## Learning Objectives

By completing this tutorial, you will be able to:

1. **Understand** governance requirements in FL
2. **Implement** model governance systems
3. **Design** approval workflows
4. **Manage** model lifecycle
5. **Deploy** governed FL pipelines

---

## Prerequisites

Before starting this tutorial, ensure you have:

- Completed tutorials 001-141
- Understanding of FL fundamentals
- Knowledge of governance practices
- Familiarity with compliance

---

## Background and Theory

### FL Governance Components

```
FL Governance:
├── Model Approval
│   ├── Review process
│   ├── Validation criteria
│   └── Sign-off workflow
├── Access Control
│   ├── Role-based permissions
│   ├── Client authorization
│   └── Data access policies
├── Audit Trail
│   ├── Training history
│   ├── Model changes
│   └── Decision logging
└── Risk Management
    ├── Model risk assessment
    ├── Bias monitoring
    └── Performance tracking
```

---

## Implementation Code

```python
#!/usr/bin/env python3
"""
Tutorial 142: FL Model Governance

This module implements model governance for FL
including approval workflows and audit trails.

Copyright 2026 Olaf Yunus Laitinen Imanov and Contributors
Released under EUPL 1.2
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime
from enum import Enum
import copy
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ApprovalStatus(Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    REVIEW = "under_review"


class Role(Enum):
    DATA_SCIENTIST = "data_scientist"
    MODEL_REVIEWER = "model_reviewer"
    GOVERNANCE_OFFICER = "governance_officer"
    ADMIN = "admin"


@dataclass
class GovConfig:
    """Governance configuration."""
    
    num_rounds: int = 30
    num_clients: int = 10
    clients_per_round: int = 5
    
    input_dim: int = 32
    hidden_dim: int = 64
    num_classes: int = 10
    
    learning_rate: float = 0.01
    batch_size: int = 32
    local_epochs: int = 3
    
    # Governance thresholds
    min_accuracy: float = 0.7
    max_bias_score: float = 0.1
    
    seed: int = 42


class AuditTrail:
    """Audit trail for governance."""
    
    def __init__(self):
        self.entries: List[Dict] = []
    
    def log(
        self,
        action: str,
        actor: str,
        details: Dict,
        model_version: Optional[str] = None
    ):
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "action": action,
            "actor": actor,
            "model_version": model_version,
            "details": details
        }
        self.entries.append(entry)
    
    def get_by_version(self, version: str) -> List[Dict]:
        return [e for e in self.entries if e.get("model_version") == version]


class ModelApproval:
    """Model approval workflow."""
    
    def __init__(self, audit: AuditTrail):
        self.audit = audit
        self.approvals: Dict[str, Dict] = {}
    
    def submit(
        self,
        version: str,
        metrics: Dict[str, float],
        submitter: str
    ) -> str:
        self.approvals[version] = {
            "status": ApprovalStatus.PENDING,
            "metrics": metrics,
            "submitter": submitter,
            "submitted_at": datetime.utcnow().isoformat(),
            "reviews": []
        }
        
        self.audit.log("SUBMIT_FOR_APPROVAL", submitter, {"metrics": metrics}, version)
        return version
    
    def review(
        self,
        version: str,
        reviewer: str,
        approved: bool,
        comments: str = ""
    ) -> None:
        if version not in self.approvals:
            return
        
        self.approvals[version]["reviews"].append({
            "reviewer": reviewer,
            "approved": approved,
            "comments": comments,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        self.audit.log(
            "REVIEW_SUBMITTED",
            reviewer,
            {"approved": approved, "comments": comments},
            version
        )
    
    def finalize(
        self,
        version: str,
        officer: str,
        min_accuracy: float
    ) -> ApprovalStatus:
        if version not in self.approvals:
            return ApprovalStatus.REJECTED
        
        approval = self.approvals[version]
        reviews = approval["reviews"]
        metrics = approval["metrics"]
        
        # Check criteria
        all_approved = all(r["approved"] for r in reviews) if reviews else False
        meets_accuracy = metrics.get("accuracy", 0) >= min_accuracy
        
        if all_approved and meets_accuracy:
            status = ApprovalStatus.APPROVED
        else:
            status = ApprovalStatus.REJECTED
        
        approval["status"] = status
        approval["finalized_by"] = officer
        approval["finalized_at"] = datetime.utcnow().isoformat()
        
        self.audit.log(
            "APPROVAL_FINALIZED",
            officer,
            {"status": status.value},
            version
        )
        
        return status


class RiskAssessment:
    """Model risk assessment."""
    
    def __init__(self, audit: AuditTrail):
        self.audit = audit
    
    def assess(
        self,
        version: str,
        metrics: Dict[str, float],
        assessor: str
    ) -> Dict[str, Any]:
        risk_level = "low"
        findings = []
        
        accuracy = metrics.get("accuracy", 0)
        if accuracy < 0.6:
            risk_level = "high"
            findings.append("Low accuracy may cause errors")
        elif accuracy < 0.75:
            risk_level = "medium"
            findings.append("Moderate accuracy")
        
        assessment = {
            "version": version,
            "risk_level": risk_level,
            "findings": findings,
            "metrics": metrics,
            "assessed_at": datetime.utcnow().isoformat()
        }
        
        self.audit.log("RISK_ASSESSMENT", assessor, assessment, version)
        
        return assessment


class GovDataset(Dataset):
    def __init__(self, n: int = 200, dim: int = 32, classes: int = 10, seed: int = 0):
        np.random.seed(seed)
        self.x = torch.randn(n, dim, dtype=torch.float32)
        self.y = torch.randint(0, classes, (n,), dtype=torch.long)
        for i in range(n):
            self.x[i, self.y[i].item() % dim] += 2.0
    
    def __len__(self): return len(self.y)
    def __getitem__(self, idx): return self.x[idx], self.y[idx]


class GovModel(nn.Module):
    def __init__(self, config: GovConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.num_classes)
        )
    
    def forward(self, x): return self.net(x)


class GovClient:
    def __init__(self, client_id: int, dataset: GovDataset, config: GovConfig):
        self.client_id = client_id
        self.dataset = dataset
        self.config = config
    
    def train(self, model: nn.Module) -> Dict:
        local = copy.deepcopy(model)
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
            "avg_loss": total_loss / num_batches
        }


class GovServer:
    """Server with governance."""
    
    def __init__(
        self,
        model: nn.Module,
        clients: List[GovClient],
        test_data: GovDataset,
        config: GovConfig
    ):
        self.model = model
        self.clients = clients
        self.test_data = test_data
        self.config = config
        
        self.audit = AuditTrail()
        self.approval = ModelApproval(self.audit)
        self.risk = RiskAssessment(self.audit)
        
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
    
    def train_with_governance(self) -> List[Dict]:
        logger.info(f"Starting governed FL")
        
        for round_num in range(self.config.num_rounds):
            version = f"v{round_num + 1}"
            
            n = min(self.config.clients_per_round, len(self.clients))
            indices = np.random.choice(len(self.clients), n, replace=False)
            selected = [self.clients[i] for i in indices]
            
            updates = [c.train(self.model) for c in selected]
            self.aggregate(updates)
            
            self.audit.log("TRAINING_ROUND", "system", {"round": round_num}, version)
            
            metrics = self.evaluate()
            
            # Governance workflow
            self.approval.submit(version, metrics, "ml_engineer")
            self.approval.review(version, "reviewer", metrics["accuracy"] > 0.6)
            status = self.approval.finalize(version, "governance_officer", self.config.min_accuracy)
            
            self.risk.assess(version, metrics, "risk_team")
            
            record = {
                "round": round_num,
                **metrics,
                "version": version,
                "status": status.value
            }
            self.history.append(record)
            
            if (round_num + 1) % 10 == 0:
                logger.info(f"Round {round_num + 1}: acc={metrics['accuracy']:.4f}, status={status.value}")
        
        return self.history


def main():
    print("=" * 60)
    print("Tutorial 142: FL Model Governance")
    print("=" * 60)
    
    config = GovConfig()
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    
    clients = [GovClient(i, GovDataset(seed=config.seed + i), config) for i in range(config.num_clients)]
    test_data = GovDataset(seed=999)
    model = GovModel(config)
    
    server = GovServer(model, clients, test_data, config)
    history = server.train_with_governance()
    
    approved = sum(1 for h in history if h["status"] == "approved")
    
    print("\n" + "=" * 60)
    print("Governance Summary")
    print(f"Total rounds: {len(history)}")
    print(f"Approved: {approved}")
    print(f"Audit entries: {len(server.audit.entries)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
```

---

## Key Insights

### Governance Best Practices

1. **Clear workflows**: Define approval process
2. **Audit everything**: Full traceability
3. **Role separation**: Different reviewers
4. **Risk assessment**: Continuous monitoring

---

## Exercises

1. **Exercise 1**: Add multi-level approval
2. **Exercise 2**: Implement bias monitoring
3. **Exercise 3**: Design dashboard
4. **Exercise 4**: Add automated gates

---

## References

1. Model Risk Management (OCC, 2011)
2. Responsible AI practices
3. MLOps governance frameworks

---

*Copyright 2026 Olaf Yunus Laitinen Imanov and Contributors. Released under EUPL 1.2.*
