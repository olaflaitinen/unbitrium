# Tutorial 114: FL for Recommendations

---

## Metadata

| Property | Value |
|----------|-------|
| **Tutorial ID** | 114 |
| **Title** | Federated Learning for Recommendation Systems |
| **Category** | Domain Applications |
| **Difficulty** | Advanced |
| **Duration** | 120 minutes |
| **Prerequisites** | Tutorial 001-113 |
| **Author** | Unbitrium Contributors |
| **Last Updated** | January 2026 |

---

## Learning Objectives

By completing this tutorial, you will be able to:

1. **Understand** FL applications in recommendation
2. **Implement** federated collaborative filtering
3. **Design** privacy-preserving recommenders
4. **Analyze** personalization vs privacy tradeoffs
5. **Deploy** FL for content recommendations

---

## Prerequisites

Before starting this tutorial, ensure you have:

- Completed tutorials 001-113
- Understanding of FL fundamentals
- Knowledge of recommendation systems
- Familiarity with matrix factorization

---

## Background and Theory

### Recommendation System Challenges

Traditional recommenders require centralized user data:
- Browsing history
- Purchase patterns
- Ratings and preferences
- Interaction logs

FL enables privacy-preserving recommendations by:
- Keeping user data on-device
- Training models locally
- Aggregating preferences securely
- Enabling personalization

### Architecture

```
FL Recommendation Architecture:
├── User Device
│   ├── Local interaction logs
│   ├── User embedding training
│   ├── On-device inference
│   └── Privacy protection
├── Content Server
│   ├── Item embeddings
│   ├── Catalog management
│   └── Content features
└── FL Coordinator
    ├── Embedding aggregation
    ├── Global model training
    └── Privacy-preserving sync
```

### Approaches

| Approach | Privacy | Accuracy | Complexity |
|----------|---------|----------|------------|
| Centralized | Low | High | Low |
| Federated | High | Medium | High |
| On-device only | Very High | Low | Low |
| Hybrid FL | High | Medium-High | Medium |

---

## Implementation Code

```python
#!/usr/bin/env python3
"""
Tutorial 114: Federated Learning for Recommendations

This module implements federated collaborative filtering
and privacy-preserving recommendation systems.

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
class RecommendationConfig:
    """Configuration for federated recommendations."""
    
    num_rounds: int = 50
    num_users: int = 100
    users_per_round: int = 20
    
    num_items: int = 500
    embedding_dim: int = 32
    hidden_dim: int = 64
    
    learning_rate: float = 0.01
    batch_size: int = 32
    local_epochs: int = 3
    
    interactions_per_user: int = 50
    
    seed: int = 42


class InteractionDataset(Dataset):
    """User-item interaction dataset."""
    
    def __init__(
        self,
        user_id: int,
        num_items: int = 500,
        n: int = 50,
        seed: int = 0
    ):
        np.random.seed(seed + user_id)
        
        self.user_id = user_id
        
        # User preferences (latent factors)
        self.user_prefs = np.random.randn(32) * 0.1
        
        # Generate interactions
        self.item_ids = []
        self.ratings = []
        
        for _ in range(n):
            item_id = np.random.randint(0, num_items)
            
            # Simulate rating based on user preference + item features
            item_features = self._get_item_features(item_id, num_items)
            rating = np.dot(self.user_prefs[:len(item_features)], item_features)
            rating = np.clip(rating + np.random.randn() * 0.5, 1, 5)
            
            self.item_ids.append(item_id)
            self.ratings.append(rating)
        
        self.item_ids = torch.tensor(self.item_ids, dtype=torch.long)
        self.ratings = torch.tensor(self.ratings, dtype=torch.float32)
    
    def _get_item_features(self, item_id: int, num_items: int) -> np.ndarray:
        """Get synthetic item features."""
        np.random.seed(item_id)
        return np.random.randn(32) * 0.1
    
    def __len__(self) -> int:
        return len(self.ratings)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            torch.tensor(self.user_id),
            self.item_ids[idx],
            self.ratings[idx]
        )


class FederatedRecommender(nn.Module):
    """
    Federated recommendation model using neural collaborative filtering.
    
    The model has two parts:
    - Item embeddings (shared globally)
    - User embeddings (kept local / federated)
    """
    
    def __init__(self, config: RecommendationConfig):
        super().__init__()
        
        self.config = config
        
        # Item embeddings (shared globally)
        self.item_embedding = nn.Embedding(
            config.num_items,
            config.embedding_dim
        )
        
        # User embedding placeholder (updated per user)
        self.user_embedding = nn.Embedding(
            config.num_users,
            config.embedding_dim
        )
        
        # Prediction MLP
        self.mlp = nn.Sequential(
            nn.Linear(config.embedding_dim * 2, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(config.hidden_dim // 2, 1)
        )
    
    def forward(
        self,
        user_ids: torch.Tensor,
        item_ids: torch.Tensor
    ) -> torch.Tensor:
        """Predict ratings."""
        user_emb = self.user_embedding(user_ids)
        item_emb = self.item_embedding(item_ids)
        
        combined = torch.cat([user_emb, item_emb], dim=-1)
        output = self.mlp(combined)
        
        return output.squeeze(-1)
    
    def get_recommendations(
        self,
        user_id: int,
        k: int = 10,
        exclude_items: Optional[List[int]] = None
    ) -> List[Tuple[int, float]]:
        """Get top-k recommendations for a user."""
        self.eval()
        
        exclude_set = set(exclude_items) if exclude_items else set()
        
        with torch.no_grad():
            user_tensor = torch.tensor([user_id])
            all_items = torch.arange(self.config.num_items)
            
            # Score all items
            users = user_tensor.expand(self.config.num_items)
            scores = self(users, all_items)
            
            # Filter and rank
            recommendations = []
            for item_id, score in enumerate(scores):
                if item_id not in exclude_set:
                    recommendations.append((item_id, score.item()))
            
            recommendations.sort(key=lambda x: x[1], reverse=True)
            return recommendations[:k]


class UserClient:
    """FL client representing a user's device."""
    
    def __init__(
        self,
        user_id: int,
        dataset: InteractionDataset,
        config: RecommendationConfig
    ):
        self.user_id = user_id
        self.dataset = dataset
        self.config = config
    
    def train(self, model: nn.Module) -> Dict[str, Any]:
        """Train on local interactions."""
        local_model = copy.deepcopy(model)
        
        # Only train user-specific and shared components
        optimizer = torch.optim.Adam(
            local_model.parameters(),
            lr=self.config.learning_rate
        )
        
        loader = DataLoader(
            self.dataset,
            batch_size=self.config.batch_size,
            shuffle=True
        )
        
        local_model.train()
        total_loss = 0.0
        num_batches = 0
        
        for _ in range(self.config.local_epochs):
            for user_ids, item_ids, ratings in loader:
                optimizer.zero_grad()
                
                predictions = local_model(user_ids, item_ids)
                loss = F.mse_loss(predictions, ratings)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(local_model.parameters(), 1.0)
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
        
        # Return only item embedding and MLP updates (privacy)
        # User embeddings stay local
        return {
            "item_embedding": local_model.item_embedding.weight.data.cpu(),
            "mlp_state": {
                k: v.cpu() for k, v in local_model.mlp.state_dict().items()
            },
            "num_samples": len(self.dataset),
            "avg_loss": total_loss / num_batches,
            "user_id": self.user_id
        }


class RecommendationServer:
    """Server for federated recommendations."""
    
    def __init__(
        self,
        model: nn.Module,
        users: List[UserClient],
        config: RecommendationConfig
    ):
        self.model = model
        self.users = users
        self.config = config
        self.history: List[Dict] = []
    
    def aggregate(self, updates: List[Dict]) -> None:
        """Aggregate item embeddings and MLP."""
        total_samples = sum(u["num_samples"] for u in updates)
        
        # Aggregate item embeddings
        new_item_emb = sum(
            (u["num_samples"] / total_samples) * u["item_embedding"].float()
            for u in updates
        )
        self.model.item_embedding.weight.data = new_item_emb
        
        # Aggregate MLP
        new_mlp_state = {}
        for key in updates[0]["mlp_state"]:
            new_mlp_state[key] = sum(
                (u["num_samples"] / total_samples) * u["mlp_state"][key].float()
                for u in updates
            )
        self.model.mlp.load_state_dict(new_mlp_state)
    
    def evaluate(
        self,
        test_users: List[UserClient],
        k: int = 10
    ) -> Dict[str, float]:
        """Evaluate recommendations."""
        self.model.eval()
        
        total_rmse = 0.0
        count = 0
        
        for user in test_users:
            loader = DataLoader(user.dataset, batch_size=64)
            
            with torch.no_grad():
                for user_ids, item_ids, ratings in loader:
                    predictions = self.model(user_ids, item_ids)
                    rmse = ((predictions - ratings) ** 2).sum().item()
                    total_rmse += rmse
                    count += len(ratings)
        
        return {
            "rmse": (total_rmse / count) ** 0.5
        }
    
    def train(self) -> List[Dict]:
        """Run federated training."""
        logger.info(f"Starting federated recommendation with {len(self.users)} users")
        
        for round_num in range(self.config.num_rounds):
            # Select users
            n = min(self.config.users_per_round, len(self.users))
            indices = np.random.choice(len(self.users), n, replace=False)
            selected = [self.users[i] for i in indices]
            
            # Collect updates
            updates = [u.train(self.model) for u in selected]
            
            # Aggregate
            self.aggregate(updates)
            
            # Evaluate
            metrics = self.evaluate(self.users[:20])
            
            record = {
                "round": round_num,
                **metrics,
                "num_users": len(selected),
                "avg_loss": np.mean([u["avg_loss"] for u in updates])
            }
            self.history.append(record)
            
            if (round_num + 1) % 10 == 0:
                logger.info(
                    f"Round {round_num + 1}: "
                    f"RMSE={metrics['rmse']:.4f}"
                )
        
        return self.history


def main():
    """Main entry point."""
    print("=" * 60)
    print("Tutorial 114: FL for Recommendations")
    print("=" * 60)
    
    config = RecommendationConfig()
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    
    # Create users
    users = []
    for i in range(config.num_users):
        dataset = InteractionDataset(
            user_id=i,
            num_items=config.num_items,
            n=config.interactions_per_user,
            seed=config.seed
        )
        client = UserClient(i, dataset, config)
        users.append(client)
    
    # Model
    model = FederatedRecommender(config)
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train
    server = RecommendationServer(model, users, config)
    history = server.train()
    
    # Get sample recommendations
    recs = model.get_recommendations(user_id=0, k=5)
    
    print("\n" + "=" * 60)
    print("Training Complete")
    print(f"Final RMSE: {history[-1]['rmse']:.4f}")
    print("\nSample recommendations for user 0:")
    for item_id, score in recs:
        print(f"  Item {item_id}: {score:.3f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
```

---

## Key Insights

### FL Recommendation Challenges

1. **Data Sparsity**: Few interactions per user
2. **Cold Start**: New users have no data
3. **Privacy**: User preferences are sensitive
4. **Personalization**: Balance global and local

### Best Practices

- Keep user embeddings local
- Share item embeddings globally
- Use differential privacy for sensitive data
- Implement on-device inference

---

## Exercises

1. **Exercise 1**: Add implicit feedback handling
2. **Exercise 2**: Implement federated matrix factorization
3. **Exercise 3**: Add content-based filtering
4. **Exercise 4**: Design cold-start solution

---

## References

1. Chen, M., et al. (2018). Federated meta-learning for recommendation. *arXiv*.
2. Ammad-ud-din, M., et al. (2019). Federated collaborative filtering. In *RecSys*.
3. Muhammad, F., et al. (2020). FedRec: Federated recommendation with explicit feedback. *IEEE TOIS*.

---

*Copyright 2026 Olaf Yunus Laitinen Imanov and Contributors. Released under EUPL 1.2.*
