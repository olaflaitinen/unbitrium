# Tutorial 138: FL Blockchain

---

## Metadata

| Property | Value |
|----------|-------|
| **Tutorial ID** | 138 |
| **Title** | Federated Learning with Blockchain |
| **Category** | Infrastructure |
| **Difficulty** | Advanced |
| **Duration** | 120 minutes |
| **Prerequisites** | Tutorial 001-137 |
| **Author** | Unbitrium Contributors |
| **Last Updated** | January 2026 |

---

## Learning Objectives

By completing this tutorial, you will be able to:

1. **Understand** blockchain-FL integration benefits
2. **Implement** decentralized FL aggregation
3. **Design** trustless FL systems
4. **Analyze** incentive mechanisms
5. **Deploy** blockchain-based FL

---

## Prerequisites

Before starting this tutorial, ensure you have:

- Completed tutorials 001-137
- Understanding of FL fundamentals
- Knowledge of blockchain basics
- Familiarity with smart contracts

---

## Background and Theory

### Why Blockchain for FL?

Blockchain addresses FL trust issues:
- No trusted central aggregator needed
- Transparent aggregation process
- Immutable training history
- Built-in incentive mechanisms

### Architecture

```
Blockchain-FL Architecture:
├── Client Layer
│   ├── Local training
│   ├── Update submission
│   └── Reward claiming
├── Blockchain Layer
│   ├── Smart contract aggregation
│   ├── Model verification
│   ├── Token management
│   └── Governance
└── Storage Layer
    ├── IPFS for models
    ├── On-chain hashes
    └── Verification proofs
```

### Trade-offs

| Aspect | Centralized FL | Blockchain FL |
|--------|---------------|---------------|
| Trust | Required | Trustless |
| Speed | Fast | Slower |
| Cost | Low | Gas fees |
| Transparency | Limited | Full |
| Scalability | High | Moderate |

---

## Implementation Code

```python
#!/usr/bin/env python3
"""
Tutorial 138: Federated Learning with Blockchain

This module implements blockchain-based FL with decentralized
aggregation and incentive mechanisms.

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
import hashlib
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class BlockchainConfig:
    """Configuration for blockchain-FL."""
    
    num_rounds: int = 30
    num_clients: int = 10
    clients_per_round: int = 5
    
    input_dim: int = 32
    hidden_dim: int = 64
    num_classes: int = 10
    
    learning_rate: float = 0.01
    batch_size: int = 32
    local_epochs: int = 3
    
    # Blockchain parameters
    stake_required: float = 100.0
    reward_per_round: float = 10.0
    validation_threshold: int = 3  # Min validators
    
    seed: int = 42


class Block:
    """A block in the FL chain."""
    
    def __init__(
        self,
        index: int,
        previous_hash: str,
        round_num: int,
        model_hash: str,
        client_contributions: Dict[int, float],
        timestamp: float
    ):
        self.index = index
        self.previous_hash = previous_hash
        self.round_num = round_num
        self.model_hash = model_hash
        self.client_contributions = client_contributions
        self.timestamp = timestamp
        
        self.hash = self.compute_hash()
    
    def compute_hash(self) -> str:
        """Compute block hash."""
        data = f"{self.index}{self.previous_hash}{self.round_num}"
        data += f"{self.model_hash}{self.client_contributions}{self.timestamp}"
        return hashlib.sha256(data.encode()).hexdigest()


class FLBlockchain:
    """Simple blockchain for FL."""
    
    def __init__(self):
        self.chain: List[Block] = []
        self.pending_updates: List[Dict] = []
        
        # Create genesis block
        self._create_genesis()
    
    def _create_genesis(self) -> None:
        """Create genesis block."""
        genesis = Block(
            index=0,
            previous_hash="0" * 64,
            round_num=-1,
            model_hash="genesis",
            client_contributions={},
            timestamp=time.time()
        )
        self.chain.append(genesis)
    
    def add_update(self, update: Dict) -> None:
        """Add pending update."""
        self.pending_updates.append(update)
    
    def create_block(
        self,
        round_num: int,
        model_hash: str,
        contributions: Dict[int, float]
    ) -> Block:
        """Create new block."""
        block = Block(
            index=len(self.chain),
            previous_hash=self.chain[-1].hash,
            round_num=round_num,
            model_hash=model_hash,
            client_contributions=contributions,
            timestamp=time.time()
        )
        self.chain.append(block)
        self.pending_updates = []
        
        return block
    
    def is_valid(self) -> bool:
        """Validate chain integrity."""
        for i in range(1, len(self.chain)):
            current = self.chain[i]
            previous = self.chain[i - 1]
            
            if current.previous_hash != previous.hash:
                return False
            if current.hash != current.compute_hash():
                return False
        
        return True


class TokenLedger:
    """Token management for incentives."""
    
    def __init__(self, num_clients: int, initial_balance: float = 1000.0):
        self.balances: Dict[int, float] = {
            i: initial_balance for i in range(num_clients)
        }
        self.staked: Dict[int, float] = {i: 0.0 for i in range(num_clients)}
        self.total_rewards: Dict[int, float] = {i: 0.0 for i in range(num_clients)}
    
    def stake(self, client_id: int, amount: float) -> bool:
        """Stake tokens to participate."""
        if self.balances[client_id] >= amount:
            self.balances[client_id] -= amount
            self.staked[client_id] += amount
            return True
        return False
    
    def unstake(self, client_id: int) -> float:
        """Unstake all tokens."""
        amount = self.staked[client_id]
        self.staked[client_id] = 0
        self.balances[client_id] += amount
        return amount
    
    def reward(self, client_id: int, amount: float) -> None:
        """Reward client."""
        self.balances[client_id] += amount
        self.total_rewards[client_id] += amount
    
    def slash(self, client_id: int, amount: float) -> None:
        """Slash stake for misbehavior."""
        slash_amount = min(amount, self.staked[client_id])
        self.staked[client_id] -= slash_amount
    
    def is_staked(self, client_id: int, min_stake: float) -> bool:
        """Check if client has minimum stake."""
        return self.staked[client_id] >= min_stake


class BlockchainDataset(Dataset):
    """Dataset for blockchain FL."""
    
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


class BlockchainModel(nn.Module):
    """Model for blockchain FL."""
    
    def __init__(self, config: BlockchainConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
    
    def compute_hash(self) -> str:
        """Compute model hash for verification."""
        hasher = hashlib.sha256()
        for param in self.parameters():
            hasher.update(param.data.numpy().tobytes())
        return hasher.hexdigest()


class BlockchainClient:
    """Client in blockchain-based FL."""
    
    def __init__(
        self,
        client_id: int,
        dataset: BlockchainDataset,
        config: BlockchainConfig,
        ledger: TokenLedger
    ):
        self.client_id = client_id
        self.dataset = dataset
        self.config = config
        self.ledger = ledger
    
    def stake(self) -> bool:
        """Stake to participate."""
        return self.ledger.stake(self.client_id, self.config.stake_required)
    
    def train(self, model: nn.Module) -> Dict[str, Any]:
        """Train and submit update."""
        if not self.ledger.is_staked(self.client_id, self.config.stake_required):
            return {"status": "not_staked", "client_id": self.client_id}
        
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
            "avg_loss": total_loss / num_batches,
            "client_id": self.client_id,
            "status": "success"
        }
    
    def validate(
        self,
        model: nn.Module,
        update: Dict
    ) -> Tuple[bool, float]:
        """Validate another client's update."""
        # Create model with update
        test_model = copy.deepcopy(model)
        test_model.load_state_dict(update["state_dict"])
        
        # Evaluate on local data
        test_model.eval()
        loader = DataLoader(self.dataset, batch_size=64)
        
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in loader:
                pred = test_model(x).argmax(dim=1)
                correct += (pred == y).sum().item()
                total += len(y)
        
        accuracy = correct / total
        is_valid = accuracy > 0.1  # Basic validity check
        
        return is_valid, accuracy


class BlockchainServer:
    """Decentralized server using blockchain."""
    
    def __init__(
        self,
        model: nn.Module,
        clients: List[BlockchainClient],
        test_data: BlockchainDataset,
        config: BlockchainConfig
    ):
        self.model = model
        self.clients = clients
        self.test_data = test_data
        self.config = config
        
        self.blockchain = FLBlockchain()
        self.ledger = clients[0].ledger  # Shared ledger
        self.history: List[Dict] = []
    
    def validate_updates(
        self,
        updates: List[Dict],
        validators: List[BlockchainClient]
    ) -> List[Dict]:
        """Validate updates using other clients."""
        valid_updates = []
        
        for update in updates:
            if update.get("status") != "success":
                continue
            
            # Get validations
            validations = []
            for validator in validators:
                if validator.client_id != update["client_id"]:
                    is_valid, acc = validator.validate(self.model, update)
                    validations.append(is_valid)
            
            # Require threshold validations
            if sum(validations) >= self.config.validation_threshold:
                valid_updates.append(update)
            else:
                # Slash stake for invalid update
                self.ledger.slash(update["client_id"], 10.0)
                logger.warning(f"Client {update['client_id']} update rejected")
        
        return valid_updates
    
    def aggregate(self, updates: List[Dict]) -> str:
        """Aggregate and record on blockchain."""
        if not updates:
            return ""
        
        total_samples = sum(u["num_samples"] for u in updates)
        new_state = {}
        
        for key in updates[0]["state_dict"]:
            new_state[key] = sum(
                (u["num_samples"] / total_samples) * u["state_dict"][key].float()
                for u in updates
            )
        
        self.model.load_state_dict(new_state)
        return self.model.compute_hash()
    
    def distribute_rewards(
        self,
        updates: List[Dict],
        round_num: int
    ) -> Dict[int, float]:
        """Distribute rewards to contributors."""
        if not updates:
            return {}
        
        total_samples = sum(u["num_samples"] for u in updates)
        contributions = {}
        
        for update in updates:
            client_id = update["client_id"]
            contribution = update["num_samples"] / total_samples
            reward = self.config.reward_per_round * contribution
            
            self.ledger.reward(client_id, reward)
            contributions[client_id] = contribution
        
        return contributions
    
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
        """Run blockchain-based FL."""
        logger.info(f"Starting blockchain FL with {len(self.clients)} clients")
        
        # All clients stake
        for client in self.clients:
            client.stake()
        
        for round_num in range(self.config.num_rounds):
            # Select clients
            n = min(self.config.clients_per_round, len(self.clients))
            indices = np.random.choice(len(self.clients), n, replace=False)
            selected = [self.clients[i] for i in indices]
            
            # Collect updates
            updates = [c.train(self.model) for c in selected]
            
            # Validate
            validators = [c for c in self.clients if c not in selected]
            valid_updates = self.validate_updates(updates, validators)
            
            # Aggregate
            model_hash = self.aggregate(valid_updates)
            
            # Distribute rewards
            contributions = self.distribute_rewards(valid_updates, round_num)
            
            # Record on blockchain
            if valid_updates:
                self.blockchain.create_block(round_num, model_hash, contributions)
            
            # Evaluate
            metrics = self.evaluate()
            
            record = {
                "round": round_num,
                **metrics,
                "valid_updates": len(valid_updates),
                "total_updates": len(updates)
            }
            self.history.append(record)
            
            if (round_num + 1) % 10 == 0:
                logger.info(
                    f"Round {round_num + 1}: acc={metrics['accuracy']:.4f}, "
                    f"chain valid={self.blockchain.is_valid()}"
                )
        
        return self.history


def main():
    """Main entry point."""
    print("=" * 60)
    print("Tutorial 138: FL with Blockchain")
    print("=" * 60)
    
    config = BlockchainConfig()
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    
    # Shared ledger
    ledger = TokenLedger(config.num_clients)
    
    # Create clients
    clients = []
    for i in range(config.num_clients):
        dataset = BlockchainDataset(client_id=i, dim=config.input_dim, seed=config.seed)
        client = BlockchainClient(i, dataset, config, ledger)
        clients.append(client)
    
    test_data = BlockchainDataset(client_id=999, n=300, seed=999)
    model = BlockchainModel(config)
    
    # Train
    server = BlockchainServer(model, clients, test_data, config)
    history = server.train()
    
    # Summary
    print("\n" + "=" * 60)
    print("Training Complete")
    print(f"Final Accuracy: {history[-1]['accuracy']:.4f}")
    print(f"Chain Length: {len(server.blockchain.chain)}")
    print(f"Chain Valid: {server.blockchain.is_valid()}")
    print("\nClient Rewards:")
    for cid, reward in ledger.total_rewards.items():
        if reward > 0:
            print(f"  Client {cid}: {reward:.2f} tokens")
    print("=" * 60)


if __name__ == "__main__":
    main()
```

---

## Key Insights

### Blockchain-FL Benefits

1. **Trustless**: No central authority needed
2. **Transparent**: All operations on-chain
3. **Incentivized**: Token rewards for participation
4. **Verifiable**: Immutable training history

### Trade-offs

- Higher latency due to consensus
- Gas costs for on-chain operations
- Storage constraints for large models

---

## Exercises

1. **Exercise 1**: Implement proof-of-stake consensus
2. **Exercise 2**: Add smart contract simulation
3. **Exercise 3**: Implement IPFS model storage
4. **Exercise 4**: Design governance mechanism

---

## References

1. Kim, H., et al. (2020). Blockchained on-device FL. *IEEE COMST*.
2. Weng, J., et al. (2021). DeepChain: Auditable FL via blockchain. *IEEE TIFS*.
3. Qu, Y., et al. (2020). Blockchain-based FL for IoT. *IEEE IoT Journal*.

---

*Copyright 2026 Olaf Yunus Laitinen Imanov and Contributors. Released under EUPL 1.2.*
