# Tutorial 139: FL Incentives

---

## Metadata

| Property | Value |
|----------|-------|
| **Tutorial ID** | 139 |
| **Title** | Federated Learning Incentive Mechanisms |
| **Category** | Economics |
| **Difficulty** | Advanced |
| **Duration** | 120 minutes |
| **Prerequisites** | Tutorial 001-138 |
| **Author** | Unbitrium Contributors |
| **Last Updated** | January 2026 |

---

## Learning Objectives

By completing this tutorial, you will be able to:

1. **Understand** incentive design in FL
2. **Implement** reward mechanisms
3. **Design** contribution-based payments
4. **Analyze** game-theoretic aspects
5. **Deploy** sustainable FL systems

---

## Prerequisites

Before starting this tutorial, ensure you have:

- Completed tutorials 001-138
- Understanding of FL fundamentals
- Knowledge of game theory basics
- Familiarity with mechanism design

---

## Background and Theory

### Why Incentives Matter

FL depends on voluntary participation:
- Clients incur costs (compute, bandwidth, energy)
- Data is valuable
- Without incentives, participation drops
- Free-rider problem exists

### Incentive Mechanisms

```
FL Incentive Mechanisms:
├── Direct Payment
│   ├── Per-round rewards
│   ├── Quality-based bonuses
│   └── Contribution scoring
├── Reputation Systems
│   ├── Participation history
│   ├── Quality scores
│   └── Reliability metrics
├── Auction Mechanisms
│   ├── First-price auction
│   ├── Reverse auction
│   └── VCG mechanism
└── Token Systems
    ├── Utility tokens
    ├── Staking rewards
    └── Governance rights
```

### Mechanism Properties

| Property | Description | Goal |
|----------|-------------|------|
| Incentive Compatible | Truth-telling optimal | Honest participation |
| Individual Rational | Non-negative utility | Voluntary join |
| Budget Balanced | Sustainable | Long-term viability |

---

## Implementation Code

```python
#!/usr/bin/env python3
"""
Tutorial 139: Federated Learning Incentives

This module implements incentive mechanisms for FL including
contribution-based rewards and reputation systems.

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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class IncentiveConfig:
    """Configuration for FL incentives."""
    
    num_rounds: int = 50
    num_clients: int = 20
    clients_per_round: int = 10
    
    input_dim: int = 32
    hidden_dim: int = 64
    num_classes: int = 10
    
    learning_rate: float = 0.01
    batch_size: int = 32
    local_epochs: int = 3
    
    # Incentive parameters
    total_budget: float = 1000.0
    base_reward: float = 5.0
    quality_bonus_multiplier: float = 2.0
    reputation_decay: float = 0.95
    
    seed: int = 42


class ReputationSystem:
    """Track and manage client reputations."""
    
    def __init__(self, num_clients: int, decay: float = 0.95):
        self.num_clients = num_clients
        self.decay = decay
        
        # Initialize reputations
        self.reputation: Dict[int, float] = {i: 0.5 for i in range(num_clients)}
        self.participation_count: Dict[int, int] = {i: 0 for i in range(num_clients)}
        self.quality_history: Dict[int, List[float]] = {i: [] for i in range(num_clients)}
    
    def update(
        self,
        client_id: int,
        quality_score: float,
        participated: bool = True
    ) -> None:
        """Update client reputation."""
        if participated:
            self.participation_count[client_id] += 1
            self.quality_history[client_id].append(quality_score)
            
            # EMA of reputation
            self.reputation[client_id] = (
                self.decay * self.reputation[client_id] +
                (1 - self.decay) * quality_score
            )
    
    def get_reputation(self, client_id: int) -> float:
        """Get client reputation."""
        return self.reputation.get(client_id, 0.5)
    
    def get_reliable_clients(
        self,
        min_reputation: float = 0.3,
        min_participations: int = 0
    ) -> List[int]:
        """Get clients above reputation threshold."""
        return [
            cid for cid in range(self.num_clients)
            if self.reputation[cid] >= min_reputation
            and self.participation_count[cid] >= min_participations
        ]


class ContributionEvaluator:
    """Evaluate client contributions."""
    
    def __init__(self, test_data):
        self.test_data = test_data
    
    def evaluate_update(
        self,
        base_model: nn.Module,
        updated_model: nn.Module
    ) -> float:
        """Evaluate contribution quality."""
        # Compare accuracy improvement
        base_acc = self._accuracy(base_model)
        updated_acc = self._accuracy(updated_model)
        
        improvement = updated_acc - base_acc
        
        # Normalize to [0, 1]
        return max(0, min(1, improvement * 10 + 0.5))
    
    def _accuracy(self, model: nn.Module) -> float:
        """Compute accuracy."""
        model.eval()
        loader = DataLoader(self.test_data, batch_size=64)
        
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in loader:
                pred = model(x).argmax(dim=1)
                correct += (pred == y).sum().item()
                total += len(y)
        
        return correct / total


class RewardDistributor:
    """Distribute rewards to clients."""
    
    def __init__(
        self,
        total_budget: float,
        base_reward: float,
        quality_multiplier: float
    ):
        self.total_budget = total_budget
        self.remaining_budget = total_budget
        self.base_reward = base_reward
        self.quality_multiplier = quality_multiplier
        
        self.total_distributed: Dict[int, float] = {}
    
    def distribute(
        self,
        contributions: Dict[int, float],
        reputations: Dict[int, float]
    ) -> Dict[int, float]:
        """Distribute rewards based on contribution and reputation."""
        rewards = {}
        
        if not contributions:
            return rewards
        
        # Calculate shares
        total_score = sum(
            contributions[cid] * (0.5 + 0.5 * reputations.get(cid, 0.5))
            for cid in contributions
        )
        
        round_budget = min(
            len(contributions) * self.base_reward * 2,
            self.remaining_budget
        )
        
        for cid, contribution in contributions.items():
            rep = reputations.get(cid, 0.5)
            score = contribution * (0.5 + 0.5 * rep)
            
            if total_score > 0:
                share = score / total_score
            else:
                share = 1.0 / len(contributions)
            
            reward = self.base_reward + share * (
                round_budget - len(contributions) * self.base_reward
            )
            reward = max(0, reward)
            
            rewards[cid] = reward
            self.total_distributed[cid] = (
                self.total_distributed.get(cid, 0) + reward
            )
        
        self.remaining_budget -= sum(rewards.values())
        
        return rewards


class IncentiveDataset(Dataset):
    """Dataset for incentive experiments."""
    
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


class IncentiveModel(nn.Module):
    """Model for incentive experiments."""
    
    def __init__(self, config: IncentiveConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class IncentiveClient:
    """Client with strategic behavior."""
    
    def __init__(
        self,
        client_id: int,
        dataset: IncentiveDataset,
        config: IncentiveConfig,
        effort_level: float = 1.0  # 0-1, how much effort client puts in
    ):
        self.client_id = client_id
        self.dataset = dataset
        self.config = config
        self.effort_level = effort_level
        
        self.total_rewards = 0.0
        self.rounds_participated = 0
    
    def decide_participation(
        self,
        expected_reward: float,
        cost: float = 1.0
    ) -> bool:
        """Decide whether to participate."""
        # Rational: participate if reward > cost
        return expected_reward > cost * (1.1 - self.effort_level * 0.1)
    
    def train(self, model: nn.Module) -> Dict[str, Any]:
        """Train based on effort level."""
        local = copy.deepcopy(model)
        
        # Effort affects training quality
        epochs = max(1, int(self.config.local_epochs * self.effort_level))
        lr = self.config.learning_rate * self.effort_level
        
        optimizer = torch.optim.SGD(local.parameters(), lr=lr)
        
        loader = DataLoader(
            self.dataset,
            batch_size=self.config.batch_size,
            shuffle=True
        )
        
        local.train()
        total_loss = 0.0
        num_batches = 0
        
        for _ in range(epochs):
            for x, y in loader:
                optimizer.zero_grad()
                loss = F.cross_entropy(local(x), y)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
        
        self.rounds_participated += 1
        
        return {
            "state_dict": {k: v.cpu() for k, v in local.state_dict().items()},
            "num_samples": len(self.dataset),
            "avg_loss": total_loss / num_batches,
            "client_id": self.client_id,
            "effort": self.effort_level
        }
    
    def receive_reward(self, amount: float) -> None:
        """Receive reward."""
        self.total_rewards += amount


class IncentiveServer:
    """Server with incentive management."""
    
    def __init__(
        self,
        model: nn.Module,
        clients: List[IncentiveClient],
        test_data: IncentiveDataset,
        config: IncentiveConfig
    ):
        self.model = model
        self.clients = clients
        self.test_data = test_data
        self.config = config
        
        self.reputation = ReputationSystem(
            len(clients),
            config.reputation_decay
        )
        self.evaluator = ContributionEvaluator(test_data)
        self.distributor = RewardDistributor(
            config.total_budget,
            config.base_reward,
            config.quality_bonus_multiplier
        )
        
        self.history: List[Dict] = []
    
    def aggregate(self, updates: List[Dict]) -> None:
        """Standard aggregation."""
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
        """Run FL with incentives."""
        logger.info(f"Starting FL with incentives, budget={self.config.total_budget}")
        
        for round_num in range(self.config.num_rounds):
            # Invite clients based on reputation
            reliable = self.reputation.get_reliable_clients(min_reputation=0.2)
            n = min(self.config.clients_per_round, len(reliable))
            
            if n < 2:
                reliable = list(range(len(self.clients)))
                n = min(self.config.clients_per_round, len(reliable))
            
            selected_ids = np.random.choice(reliable, n, replace=False).tolist()
            
            # Collect updates
            updates = []
            contributions = {}
            
            base_model = copy.deepcopy(self.model)
            
            for cid in selected_ids:
                client = self.clients[cid]
                
                # Client decides to participate
                expected_reward = self.config.base_reward * (
                    1 + self.reputation.get_reputation(cid)
                )
                
                if not client.decide_participation(expected_reward):
                    continue
                
                update = client.train(self.model)
                updates.append(update)
                
                # Evaluate contribution
                updated_model = IncentiveModel(self.config)
                updated_model.load_state_dict(update["state_dict"])
                quality = self.evaluator.evaluate_update(base_model, updated_model)
                
                contributions[cid] = quality
                self.reputation.update(cid, quality, True)
            
            # Aggregate
            self.aggregate(updates)
            
            # Distribute rewards
            reputations = {cid: self.reputation.get_reputation(cid) for cid in contributions}
            rewards = self.distributor.distribute(contributions, reputations)
            
            for cid, reward in rewards.items():
                self.clients[cid].receive_reward(reward)
            
            # Evaluate
            metrics = self.evaluate()
            
            record = {
                "round": round_num,
                **metrics,
                "participants": len(updates),
                "total_rewards": sum(rewards.values()),
                "remaining_budget": self.distributor.remaining_budget
            }
            self.history.append(record)
            
            if (round_num + 1) % 10 == 0:
                logger.info(
                    f"Round {round_num + 1}: acc={metrics['accuracy']:.4f}, "
                    f"participants={len(updates)}"
                )
        
        return self.history


def main():
    """Main entry point."""
    print("=" * 60)
    print("Tutorial 139: FL Incentives")
    print("=" * 60)
    
    config = IncentiveConfig()
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    
    # Create clients with varying effort levels
    clients = []
    for i in range(config.num_clients):
        effort = np.random.uniform(0.3, 1.0)
        dataset = IncentiveDataset(client_id=i, dim=config.input_dim, seed=config.seed)
        client = IncentiveClient(i, dataset, config, effort)
        clients.append(client)
    
    test_data = IncentiveDataset(client_id=999, n=300, seed=999)
    model = IncentiveModel(config)
    
    # Train
    server = IncentiveServer(model, clients, test_data, config)
    history = server.train()
    
    # Summary
    print("\n" + "=" * 60)
    print("Training Complete")
    print(f"Final Accuracy: {history[-1]['accuracy']:.4f}")
    print(f"Remaining Budget: {server.distributor.remaining_budget:.2f}")
    print("\nTop Rewarded Clients:")
    
    sorted_clients = sorted(
        clients,
        key=lambda c: c.total_rewards,
        reverse=True
    )[:5]
    
    for c in sorted_clients:
        print(f"  Client {c.client_id}: {c.total_rewards:.2f} (effort={c.effort_level:.2f})")
    
    print("=" * 60)


if __name__ == "__main__":
    main()
```

---

## Key Insights

### Incentive Design Principles

1. **Reward quality**: Not just participation
2. **Build reputation**: Long-term relationships
3. **Budget management**: Sustainable payouts
4. **Prevent gaming**: Robust mechanisms

---

## Exercises

1. **Exercise 1**: Implement auction-based selection
2. **Exercise 2**: Add penalty for low-quality updates
3. **Exercise 3**: Design token economy
4. **Exercise 4**: Model strategic client behavior

---

## References

1. Kang, J., et al. (2019). Incentive mechanism for FL. *IEEE TMC*.
2. Zhan, Y., et al. (2020). Incentive mechanism design for FL. *IEEE TPDS*.
3. Pandey, S., et al. (2020). A crowdsourcing framework for FL. In *INFOCOM*.

---

*Copyright 2026 Olaf Yunus Laitinen Imanov and Contributors. Released under EUPL 1.2.*
