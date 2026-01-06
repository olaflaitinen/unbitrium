# Tutorial 141: FL Regulatory Compliance

---

## Metadata

| Property | Value |
|----------|-------|
| **Tutorial ID** | 141 |
| **Title** | Federated Learning Regulatory Compliance |
| **Category** | Governance |
| **Difficulty** | Advanced |
| **Duration** | 120 minutes |
| **Prerequisites** | Tutorial 001-140 |
| **Author** | Unbitrium Contributors |
| **Last Updated** | January 2026 |

---

## Learning Objectives

By completing this tutorial, you will be able to:

1. **Understand** regulatory requirements for FL
2. **Implement** compliance-aware FL systems
3. **Design** auditable training pipelines
4. **Analyze** privacy regulation implications
5. **Deploy** compliant FL solutions

---

## Prerequisites

Before starting this tutorial, ensure you have:

- Completed tutorials 001-140
- Understanding of FL fundamentals
- Knowledge of data privacy regulations
- Familiarity with compliance frameworks

---

## Background and Theory

### Key Regulations

```
Regulatory Landscape:
├── Data Privacy
│   ├── GDPR (EU)
│   ├── CCPA (California)
│   ├── LGPD (Brazil)
│   └── PDPA (Singapore)
├── Sector-Specific
│   ├── HIPAA (Healthcare)
│   ├── PCI-DSS (Finance)
│   ├── SOX (Corporate)
│   └── FERPA (Education)
├── AI Regulations
│   ├── EU AI Act
│   ├── FDA AI/ML Guidelines
│   └── National AI Strategies
└── Cross-Border
    ├── Data localization
    ├── Transfer mechanisms
    └── Adequacy decisions
```

### GDPR Requirements for FL

| Requirement | FL Implementation |
|-------------|-------------------|
| Lawful Basis | Consent or legitimate interest |
| Data Minimization | Local training, only share updates |
| Purpose Limitation | Clearly defined model purpose |
| Right to Erasure | Client can leave, unlearning |
| Accountability | Audit logs, documentation |

---

## Implementation Code

```python
#!/usr/bin/env python3
"""
Tutorial 141: FL Regulatory Compliance

This module implements compliance-aware FL with audit logging,
consent management, and privacy-preserving mechanisms.

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
import json
import hashlib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ComplianceConfig:
    """Configuration for compliant FL."""

    num_rounds: int = 30
    num_clients: int = 15
    clients_per_round: int = 8

    input_dim: int = 32
    hidden_dim: int = 64
    num_classes: int = 10

    learning_rate: float = 0.01
    batch_size: int = 32
    local_epochs: int = 3

    # Compliance parameters
    require_consent: bool = True
    enable_audit: bool = True
    data_retention_days: int = 90
    enable_dp: bool = True
    dp_epsilon: float = 1.0

    seed: int = 42


class AuditLog:
    """Audit log for FL operations."""

    def __init__(self):
        self.entries: List[Dict] = []

    def log(
        self,
        event_type: str,
        client_id: Optional[int],
        details: Dict,
        timestamp: Optional[datetime] = None
    ) -> str:
        """Log an event."""
        if timestamp is None:
            timestamp = datetime.utcnow()

        entry = {
            "id": self._generate_id(),
            "timestamp": timestamp.isoformat(),
            "event_type": event_type,
            "client_id": client_id,
            "details": details
        }

        self.entries.append(entry)
        return entry["id"]

    def _generate_id(self) -> str:
        """Generate unique log ID."""
        content = f"{len(self.entries)}-{datetime.utcnow().isoformat()}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def get_client_events(self, client_id: int) -> List[Dict]:
        """Get all events for a client."""
        return [e for e in self.entries if e.get("client_id") == client_id]

    def export(self) -> str:
        """Export log as JSON."""
        return json.dumps(self.entries, indent=2)


class ConsentManager:
    """Manage client consent for FL participation."""

    def __init__(self, audit_log: AuditLog):
        self.consents: Dict[int, Dict] = {}
        self.audit_log = audit_log

    def request_consent(
        self,
        client_id: int,
        purposes: List[str],
        data_categories: List[str]
    ) -> bool:
        """Request consent from client (simulated)."""
        # In practice, this would be a UI interaction
        consent_given = True  # Simulated

        if consent_given:
            self.consents[client_id] = {
                "granted": datetime.utcnow().isoformat(),
                "purposes": purposes,
                "data_categories": data_categories,
                "valid": True
            }

            self.audit_log.log(
                "CONSENT_GRANTED",
                client_id,
                {"purposes": purposes, "data_categories": data_categories}
            )

        return consent_given

    def has_valid_consent(self, client_id: int) -> bool:
        """Check if client has valid consent."""
        consent = self.consents.get(client_id)
        return consent is not None and consent.get("valid", False)

    def revoke_consent(self, client_id: int) -> None:
        """Revoke client consent."""
        if client_id in self.consents:
            self.consents[client_id]["valid"] = False
            self.consents[client_id]["revoked"] = datetime.utcnow().isoformat()

            self.audit_log.log(
                "CONSENT_REVOKED",
                client_id,
                {}
            )


class DataProtectionOfficer:
    """DPO functions for FL."""

    def __init__(self, audit_log: AuditLog):
        self.audit_log = audit_log
        self.data_access_requests: List[Dict] = []
        self.erasure_requests: List[Dict] = []

    def handle_access_request(
        self,
        client_id: int
    ) -> Dict[str, Any]:
        """Handle GDPR Article 15 access request."""
        request = {
            "id": len(self.data_access_requests),
            "client_id": client_id,
            "timestamp": datetime.utcnow().isoformat(),
            "status": "completed"
        }

        # Compile data about client
        client_events = self.audit_log.get_client_events(client_id)

        response = {
            "participation_history": client_events,
            "data_categories": ["model_updates", "training_metadata"],
            "purposes": ["federated_model_training"],
            "recipients": ["fl_coordinator"]
        }

        request["response"] = response
        self.data_access_requests.append(request)

        self.audit_log.log(
            "ACCESS_REQUEST_COMPLETED",
            client_id,
            {"request_id": request["id"]}
        )

        return response

    def handle_erasure_request(
        self,
        client_id: int
    ) -> Dict[str, Any]:
        """Handle GDPR Article 17 erasure request."""
        request = {
            "id": len(self.erasure_requests),
            "client_id": client_id,
            "timestamp": datetime.utcnow().isoformat(),
            "status": "processing"
        }

        # Note: Full erasure in FL is complex (model unlearning)
        # Here we mark for future unlearning

        request["actions"] = [
            "client_data_marked_for_exclusion",
            "model_unlearning_scheduled"
        ]
        request["status"] = "completed"

        self.erasure_requests.append(request)

        self.audit_log.log(
            "ERASURE_REQUEST_COMPLETED",
            client_id,
            {"request_id": request["id"]}
        )

        return request


class ComplianceDataset(Dataset):
    """Dataset with compliance metadata."""

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

        # Compliance metadata
        self.collection_date = datetime.utcnow()
        self.legal_basis = "consent"

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx]


class ComplianceModel(nn.Module):
    """Model with compliance metadata."""

    def __init__(self, config: ComplianceConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.num_classes)
        )

        # Model card metadata
        self.metadata = {
            "version": "1.0.0",
            "purpose": "classification",
            "training_method": "federated_learning",
            "dp_enabled": config.enable_dp,
            "dp_epsilon": config.dp_epsilon if config.enable_dp else None
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ComplianceClient:
    """Compliant FL client."""

    def __init__(
        self,
        client_id: int,
        dataset: ComplianceDataset,
        config: ComplianceConfig
    ):
        self.client_id = client_id
        self.dataset = dataset
        self.config = config
        self.consent_given = False

    def give_consent(self) -> None:
        """Record consent."""
        self.consent_given = True

    def train(self, model: nn.Module) -> Optional[Dict[str, Any]]:
        """Train with compliance checks."""
        if self.config.require_consent and not self.consent_given:
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

                # Apply DP if enabled
                if self.config.enable_dp:
                    self._clip_and_noise_gradients(local)

                optimizer.step()

                total_loss += loss.item()
                num_batches += 1

        return {
            "state_dict": {k: v.cpu() for k, v in local.state_dict().items()},
            "num_samples": len(self.dataset),
            "avg_loss": total_loss / num_batches,
            "client_id": self.client_id,
            "dp_applied": self.config.enable_dp
        }

    def _clip_and_noise_gradients(
        self,
        model: nn.Module,
        clip_norm: float = 1.0
    ) -> None:
        """Apply DP gradient clipping and noise."""
        total_norm = 0.0
        for param in model.parameters():
            if param.grad is not None:
                total_norm += param.grad.data.norm() ** 2
        total_norm = total_norm ** 0.5

        clip_factor = min(1.0, clip_norm / (total_norm + 1e-8))

        noise_scale = clip_norm * np.sqrt(2 * np.log(1.25 / 0.1)) / self.config.dp_epsilon

        for param in model.parameters():
            if param.grad is not None:
                param.grad.data.mul_(clip_factor)
                param.grad.data.add_(torch.randn_like(param.grad) * noise_scale)


class ComplianceServer:
    """Server with compliance management."""

    def __init__(
        self,
        model: nn.Module,
        clients: List[ComplianceClient],
        test_data: ComplianceDataset,
        config: ComplianceConfig
    ):
        self.model = model
        self.clients = clients
        self.test_data = test_data
        self.config = config

        # Compliance components
        self.audit_log = AuditLog()
        self.consent_manager = ConsentManager(self.audit_log)
        self.dpo = DataProtectionOfficer(self.audit_log)

        self.history: List[Dict] = []

    def setup_compliance(self) -> None:
        """Initialize compliance for all clients."""
        purposes = ["model_improvement", "service_enhancement"]
        categories = ["model_updates", "training_metadata"]

        for client in self.clients:
            if self.consent_manager.request_consent(
                client.client_id, purposes, categories
            ):
                client.give_consent()

    def aggregate(self, updates: List[Dict]) -> None:
        """Aggregate with audit."""
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

        # Audit
        self.audit_log.log(
            "MODEL_AGGREGATION",
            None,
            {
                "num_contributors": len(updates),
                "client_ids": [u["client_id"] for u in updates]
            }
        )

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
        """Run compliant FL training."""
        self.setup_compliance()

        logger.info(f"Starting compliant FL with {len(self.clients)} clients")

        for round_num in range(self.config.num_rounds):
            # Select only consenting clients
            eligible = [
                c for c in self.clients
                if self.consent_manager.has_valid_consent(c.client_id)
            ]

            n = min(self.config.clients_per_round, len(eligible))
            indices = np.random.choice(len(eligible), n, replace=False)
            selected = [eligible[i] for i in indices]

            # Log selection
            self.audit_log.log(
                "ROUND_STARTED",
                None,
                {
                    "round": round_num,
                    "selected_clients": [c.client_id for c in selected]
                }
            )

            # Collect updates
            updates = []
            for client in selected:
                update = client.train(self.model)
                if update:
                    updates.append(update)

                    self.audit_log.log(
                        "CLIENT_UPDATE_RECEIVED",
                        client.client_id,
                        {"round": round_num, "dp_applied": update.get("dp_applied", False)}
                    )

            # Aggregate
            self.aggregate(updates)

            # Evaluate
            metrics = self.evaluate()

            record = {
                "round": round_num,
                **metrics,
                "participants": len(updates),
                "audit_entries": len(self.audit_log.entries)
            }
            self.history.append(record)

            if (round_num + 1) % 10 == 0:
                logger.info(
                    f"Round {round_num + 1}: acc={metrics['accuracy']:.4f}"
                )

        return self.history


def main():
    """Main entry point."""
    print("=" * 60)
    print("Tutorial 141: FL Regulatory Compliance")
    print("=" * 60)

    config = ComplianceConfig()
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    # Create clients
    clients = []
    for i in range(config.num_clients):
        dataset = ComplianceDataset(client_id=i, dim=config.input_dim, seed=config.seed)
        client = ComplianceClient(i, dataset, config)
        clients.append(client)

    test_data = ComplianceDataset(client_id=999, n=300, seed=999)
    model = ComplianceModel(config)

    # Train
    server = ComplianceServer(model, clients, test_data, config)
    history = server.train()

    # Summary
    print("\n" + "=" * 60)
    print("Training Complete")
    print(f"Final Accuracy: {history[-1]['accuracy']:.4f}")
    print(f"Total Audit Entries: {len(server.audit_log.entries)}")
    print(f"DP Enabled: {config.enable_dp}")
    print(f"DP Epsilon: {config.dp_epsilon}")
    print("=" * 60)


if __name__ == "__main__":
    main()
```

---

## Key Insights

### Compliance Best Practices

1. **Document everything**: Audit logs
2. **Get consent**: Before training
3. **Enable DP**: Privacy by design
4. **Support rights**: Access, erasure

---

## Exercises

1. **Exercise 1**: Implement data portability
2. **Exercise 2**: Add machine unlearning
3. **Exercise 3**: Create compliance dashboard
4. **Exercise 4**: Implement cross-border checks

---

## References

1. GDPR (2016). General Data Protection Regulation.
2. EU AI Act (2024). Artificial Intelligence Act.
3. Truong, N., et al. (2021). Privacy preservation in FL: A survey. *IEEE Access*.

---

*Copyright 2026 Olaf Yunus Laitinen Imanov and Contributors. Released under EUPL 1.2.*
