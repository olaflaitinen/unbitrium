# Tutorial 098: FL Trusted Execution Environments

---

## Metadata

| Property | Value |
|----------|-------|
| **Tutorial ID** | 098 |
| **Title** | FL Trusted Execution Environments |
| **Category** | Privacy |
| **Difficulty** | Expert |
| **Duration** | 120 minutes |
| **Prerequisites** | Tutorial 001-097 |
| **Author** | Unbitrium Contributors |
| **Last Updated** | January 2026 |

---

## Learning Objectives

By completing this tutorial, you will be able to:

1. **Understand** TEE concepts
2. **Implement** TEE-based aggregation
3. **Design** secure enclaves for FL
4. **Analyze** TEE security guarantees
5. **Deploy** TEE-enabled FL

---

## Background and Theory

### TEE Architecture

```
TEE Components:
├── Secure Enclaves
│   ├── Intel SGX
│   ├── ARM TrustZone
│   └── AMD SEV
├── Attestation
│   ├── Local attestation
│   ├── Remote attestation
│   └── Quote verification
├── FL Integration
│   ├── Aggregation in enclave
│   ├── Secure computation
│   └── Key management
└── Trust Model
    ├── Trusted computing base
    ├── Side-channel protection
    └── Memory encryption
```

---

## Implementation Code

```python
#!/usr/bin/env python3
"""
Tutorial 098: FL Trusted Execution Environments

This module simulates TEE-based secure aggregation
for federated learning.

Copyright 2026 Olaf Yunus Laitinen Imanov and Contributors
Released under EUPL 1.2
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass
from typing import Dict, List, Optional
import copy
import hashlib
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TEEConfig:
    """TEE configuration."""
    
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


class Enclave:
    """Simulated secure enclave."""
    
    def __init__(self, enclave_id: str):
        self.enclave_id = enclave_id
        self.sealed_data: Dict[str, bytes] = {}
        self._measurement = self._compute_measurement()
    
    def _compute_measurement(self) -> str:
        """Compute enclave measurement."""
        code = f"enclave_{self.enclave_id}_code"
        return hashlib.sha256(code.encode()).hexdigest()[:32]
    
    def get_quote(self) -> Dict[str, str]:
        """Generate attestation quote."""
        return {
            "enclave_id": self.enclave_id,
            "measurement": self._measurement,
            "signature": hashlib.sha256(
                f"{self.enclave_id}:{self._measurement}".encode()
            ).hexdigest()[:16]
        }
    
    def seal(self, key: str, data: torch.Tensor) -> None:
        """Seal data to enclave."""
        serialized = data.numpy().tobytes()
        encrypted = hashlib.sha256(serialized + self.enclave_id.encode()).digest()
        self.sealed_data[key] = serialized
    
    def unseal(self, key: str, shape: tuple) -> Optional[torch.Tensor]:
        """Unseal data from enclave."""
        if key not in self.sealed_data:
            return None
        
        data = np.frombuffer(self.sealed_data[key], dtype=np.float32)
        return torch.from_numpy(data.copy()).reshape(shape)


class SecureAggregator(Enclave):
    """Secure aggregator in TEE."""
    
    def __init__(self):
        super().__init__("aggregator")
        self.pending_updates: List[Dict[str, torch.Tensor]] = []
    
    def verify_client(self, quote: Dict[str, str]) -> bool:
        """Verify client attestation."""
        expected_sig = hashlib.sha256(
            f"{quote['enclave_id']}:{quote['measurement']}".encode()
        ).hexdigest()[:16]
        return quote["signature"] == expected_sig
    
    def receive_update(
        self,
        update: Dict[str, torch.Tensor],
        quote: Dict[str, str]
    ) -> bool:
        """Receive update in enclave."""
        if not self.verify_client(quote):
            logger.warning("Client attestation failed")
            return False
        
        self.pending_updates.append(update)
        return True
    
    def aggregate(self) -> Dict[str, torch.Tensor]:
        """Aggregate in enclave."""
        if not self.pending_updates:
            return {}
        
        n = len(self.pending_updates)
        aggregated = {}
        
        for key in self.pending_updates[0]:
            aggregated[key] = sum(u[key] for u in self.pending_updates) / n
        
        self.pending_updates.clear()
        return aggregated


class TEEDataset(Dataset):
    def __init__(self, n: int = 200, dim: int = 32, classes: int = 10, seed: int = 0):
        np.random.seed(seed)
        self.x = torch.randn(n, dim, dtype=torch.float32)
        self.y = torch.randint(0, classes, (n,), dtype=torch.long)
        for i in range(n):
            self.x[i, self.y[i].item() % dim] += 2.0
    
    def __len__(self): return len(self.y)
    def __getitem__(self, idx): return self.x[idx], self.y[idx]


class TEEModel(nn.Module):
    def __init__(self, config: TEEConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.num_classes)
        )
    
    def forward(self, x): return self.net(x)


class TEEClient:
    """Client with TEE."""
    
    def __init__(self, client_id: int, dataset: TEEDataset, config: TEEConfig):
        self.client_id = client_id
        self.dataset = dataset
        self.config = config
        
        self.enclave = Enclave(f"client_{client_id}")
    
    def train(self, model: nn.Module) -> Dict:
        local = copy.deepcopy(model)
        optimizer = torch.optim.SGD(local.parameters(), lr=self.config.learning_rate)
        loader = DataLoader(self.dataset, batch_size=self.config.batch_size, shuffle=True)
        
        local.train()
        for _ in range(self.config.local_epochs):
            for x, y in loader:
                optimizer.zero_grad()
                loss = F.cross_entropy(local(x), y)
                loss.backward()
                optimizer.step()
        
        return {
            "state_dict": {k: v.cpu() for k, v in local.state_dict().items()},
            "quote": self.enclave.get_quote(),
            "num_samples": len(self.dataset)
        }


class TEEServer:
    """Server with TEE aggregation."""
    
    def __init__(self, model: nn.Module, clients: List[TEEClient], test_data: TEEDataset, config: TEEConfig):
        self.model = model
        self.clients = clients
        self.test_data = test_data
        self.config = config
        
        self.aggregator = SecureAggregator()
        self.history: List[Dict] = []
    
    def aggregate(self, updates: List[Dict]) -> None:
        # Verify and receive in enclave
        for u in updates:
            self.aggregator.receive_update(u["state_dict"], u["quote"])
        
        # Aggregate in enclave
        aggregated = self.aggregator.aggregate()
        self.model.load_state_dict(aggregated)
    
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
        logger.info("Starting TEE-based FL")
        
        for round_num in range(self.config.num_rounds):
            n = min(self.config.clients_per_round, len(self.clients))
            indices = np.random.choice(len(self.clients), n, replace=False)
            selected = [self.clients[i] for i in indices]
            
            updates = [c.train(self.model) for c in selected]
            self.aggregate(updates)
            
            metrics = self.evaluate()
            
            record = {"round": round_num, **metrics}
            self.history.append(record)
            
            if (round_num + 1) % 10 == 0:
                logger.info(f"Round {round_num + 1}: acc={metrics['accuracy']:.4f}")
        
        return self.history


def main():
    print("=" * 60)
    print("Tutorial 098: FL Trusted Execution Environments")
    print("=" * 60)
    
    config = TEEConfig()
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    
    clients = [TEEClient(i, TEEDataset(seed=config.seed + i), config) for i in range(config.num_clients)]
    test_data = TEEDataset(seed=999)
    model = TEEModel(config)
    
    server = TEEServer(model, clients, test_data, config)
    history = server.train()
    
    print("\n" + "=" * 60)
    print("TEE Training Complete")
    print(f"Final accuracy: {history[-1]['accuracy']:.4f}")
    print(f"Aggregator quote: {server.aggregator.get_quote()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
```

---

## Key Insights

### TEE Best Practices

1. **Remote attestation**: Verify enclave integrity
2. **Minimal TCB**: Keep trusted code small
3. **Side-channel aware**: Consider timing attacks
4. **Key management**: Secure key provisioning

---

## Exercises

1. **Exercise 1**: Add remote attestation
2. **Exercise 2**: Implement sealed storage
3. **Exercise 3**: Design enclave failure handling
4. **Exercise 4**: Add side-channel mitigations

---

## References

1. Intel SGX Developer Reference
2. Costan, V., & Devadas, S. (2016). Intel SGX explained. *IACR ePrint*.
3. Mo, F., et al. (2021). PPFL: Privacy-preserving FL with TEEs. In *MobiSys*.

---

*Copyright 2026 Olaf Yunus Laitinen Imanov and Contributors. Released under EUPL 1.2.*
