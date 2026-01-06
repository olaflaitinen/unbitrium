# Tutorial 099: FL Homomorphic Encryption

---

## Metadata

| Property | Value |
|----------|-------|
| **Tutorial ID** | 099 |
| **Title** | FL Homomorphic Encryption |
| **Category** | Privacy |
| **Difficulty** | Expert |
| **Duration** | 120 minutes |
| **Prerequisites** | Tutorial 001-098 |
| **Author** | Unbitrium Contributors |
| **Last Updated** | January 2026 |

---

## Learning Objectives

By completing this tutorial, you will be able to:

1. **Understand** homomorphic encryption
2. **Implement** HE-based aggregation
3. **Design** encrypted FL protocols
4. **Analyze** HE performance
5. **Deploy** encrypted FL systems

---

## Background and Theory

### HE Concepts

```
Homomorphic Encryption:
├── Types
│   ├── Partially HE (add or multiply)
│   ├── Somewhat HE (limited ops)
│   └── Fully HE (arbitrary ops)
├── Schemes
│   ├── Paillier (additive)
│   ├── CKKS (approximate)
│   └── BFV (exact integers)
├── Operations
│   ├── Encrypted addition
│   ├── Encrypted multiplication
│   └── Bootstrapping
└── FL Integration
    ├── Encrypted gradients
    ├── Aggregation on ciphertexts
    └── Decryption at server
```

---

## Implementation Code

```python
#!/usr/bin/env python3
"""
Tutorial 099: FL Homomorphic Encryption

This module simulates HE-based federated learning
with encrypted aggregation.

Copyright 2026 Olaf Yunus Laitinen Imanov and Contributors
Released under EUPL 1.2
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass
from typing import Dict, List, Tuple
import copy
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class HEConfig:
    """HE configuration."""
    
    num_rounds: int = 30
    num_clients: int = 10
    clients_per_round: int = 5
    
    input_dim: int = 32
    hidden_dim: int = 64
    num_classes: int = 10
    
    learning_rate: float = 0.01
    batch_size: int = 32
    local_epochs: int = 3
    
    # HE params (simulated)
    precision: int = 16
    
    seed: int = 42


class SimulatedHE:
    """Simulated homomorphic encryption."""
    
    def __init__(self, precision: int = 16):
        self.precision = precision
        self.scale = 2 ** precision
        self.public_key = np.random.randint(0, 1000000)
        self.secret_key = np.random.randint(0, 1000000)
    
    def encrypt(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """Encrypt tensor (simulated)."""
        # Quantize to fixed-point
        quantized = (tensor * self.scale).round()
        
        # Add encryption noise (simulated)
        noise = torch.randn_like(quantized) * 0.01
        ciphertext = quantized + noise
        
        metadata = {
            "shape": tensor.shape,
            "scale": self.scale,
            "original_dtype": tensor.dtype
        }
        
        return ciphertext, metadata
    
    def decrypt(self, ciphertext: torch.Tensor, metadata: Dict) -> torch.Tensor:
        """Decrypt tensor (simulated)."""
        # Remove scale
        decrypted = ciphertext / metadata["scale"]
        return decrypted.to(metadata["original_dtype"])
    
    def add_encrypted(
        self,
        ct1: Tuple[torch.Tensor, Dict],
        ct2: Tuple[torch.Tensor, Dict]
    ) -> Tuple[torch.Tensor, Dict]:
        """Add two ciphertexts."""
        c1, m1 = ct1
        c2, m2 = ct2
        
        result = c1 + c2
        metadata = {
            "shape": m1["shape"],
            "scale": m1["scale"],
            "original_dtype": m1["original_dtype"]
        }
        
        return result, metadata
    
    def scalar_mult_encrypted(
        self,
        ct: Tuple[torch.Tensor, Dict],
        scalar: float
    ) -> Tuple[torch.Tensor, Dict]:
        """Multiply ciphertext by scalar."""
        c, m = ct
        
        result = c * scalar
        return result, m


class HEDataset(Dataset):
    def __init__(self, n: int = 200, dim: int = 32, classes: int = 10, seed: int = 0):
        np.random.seed(seed)
        self.x = torch.randn(n, dim, dtype=torch.float32)
        self.y = torch.randint(0, classes, (n,), dtype=torch.long)
        for i in range(n):
            self.x[i, self.y[i].item() % dim] += 2.0
    
    def __len__(self): return len(self.y)
    def __getitem__(self, idx): return self.x[idx], self.y[idx]


class HEModel(nn.Module):
    def __init__(self, config: HEConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.num_classes)
        )
    
    def forward(self, x): return self.net(x)


class HEClient:
    """Client with HE."""
    
    def __init__(self, client_id: int, dataset: HEDataset, config: HEConfig, he: SimulatedHE):
        self.client_id = client_id
        self.dataset = dataset
        self.config = config
        self.he = he
    
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
        
        # Encrypt model update
        encrypted = {}
        for k, v in local.state_dict().items():
            encrypted[k] = self.he.encrypt(v.cpu())
        
        return {
            "encrypted_state": encrypted,
            "num_samples": len(self.dataset)
        }


class HEServer:
    """Server with HE aggregation."""
    
    def __init__(self, model: nn.Module, clients: List[HEClient], test_data: HEDataset, config: HEConfig, he: SimulatedHE):
        self.model = model
        self.clients = clients
        self.test_data = test_data
        self.config = config
        self.he = he
        self.history: List[Dict] = []
    
    def aggregate(self, updates: List[Dict]) -> None:
        n = len(updates)
        
        # Sum encrypted states
        aggregated = {}
        for key in updates[0]["encrypted_state"]:
            result = updates[0]["encrypted_state"][key]
            
            for u in updates[1:]:
                result = self.he.add_encrypted(result, u["encrypted_state"][key])
            
            # Divide by n
            result = self.he.scalar_mult_encrypted(result, 1.0 / n)
            
            # Decrypt
            aggregated[key] = self.he.decrypt(*result)
        
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
        logger.info("Starting HE-FL")
        
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
    print("Tutorial 099: FL Homomorphic Encryption")
    print("=" * 60)
    
    config = HEConfig()
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    
    he = SimulatedHE(config.precision)
    
    clients = [HEClient(i, HEDataset(seed=config.seed + i), config, he) for i in range(config.num_clients)]
    test_data = HEDataset(seed=999)
    model = HEModel(config)
    
    server = HEServer(model, clients, test_data, config, he)
    history = server.train()
    
    print("\n" + "=" * 60)
    print("HE Training Complete")
    print(f"Final accuracy: {history[-1]['accuracy']:.4f}")
    print(f"Precision: {config.precision} bits")
    print("=" * 60)


if __name__ == "__main__":
    main()
```

---

## Key Insights

### HE Best Practices

1. **Choose scheme**: Match to operations needed
2. **Manage noise**: Bootstrapping when needed
3. **Quantization**: Fixed-point for HE
4. **Batching**: Pack values for efficiency

---

## Exercises

1. **Exercise 1**: Implement CKKS encoding
2. **Exercise 2**: Add bootstrapping
3. **Exercise 3**: Design batched HE
4. **Exercise 4**: Benchmark performance

---

## References

1. Gentry, C. (2009). Fully homomorphic encryption using ideal lattices. In *STOC*.
2. Cheon, J.H., et al. (2017). Homomorphic encryption for arithmetic of approximate numbers. In *ASIACRYPT*.
3. Bos, J.W., et al. (2021). PALISADE: An open-source HE library.

---

*Copyright 2026 Olaf Yunus Laitinen Imanov and Contributors. Released under EUPL 1.2.*
