# Tutorial 113: FL with RL

---

## Metadata

| Property | Value |
|----------|-------|
| **Tutorial ID** | 113 |
| **Title** | Federated Learning with Reinforcement Learning |
| **Category** | Advanced Architectures |
| **Difficulty** | Expert |
| **Duration** | 120 minutes |
| **Prerequisites** | Tutorial 001-112 |
| **Author** | Unbitrium Contributors |
| **Last Updated** | January 2026 |

---

## Learning Objectives

By completing this tutorial, you will be able to:

1. **Understand** FL applications in RL
2. **Implement** federated policy learning
3. **Design** distributed RL systems
4. **Analyze** experience sharing strategies
5. **Deploy** federated RL agents

---

## Prerequisites

Before starting this tutorial, ensure you have:

- Completed tutorials 001-112
- Understanding of FL fundamentals
- Knowledge of reinforcement learning
- Familiarity with policy gradients

---

## Background and Theory

### FL + RL Challenges

Combining FL and RL presents unique challenges:
- Different environments per client
- Non-stationary data distributions
- Credit assignment across clients
- Diverse exploration strategies

### Federated RL Architectures

```
Federated RL:
├── Policy Aggregation
│   ├── Average policy networks
│   ├── Weighted by rewards
│   └── Trust-region updates
├── Experience Sharing
│   ├── Trajectory sharing
│   ├── Gradient sharing
│   └── Model distillation
└── Personalization
    ├── Local adaptation
    ├── Meta-learning
    └── Hierarchical policies
```

---

## Implementation Code

```python
#!/usr/bin/env python3
"""
Tutorial 113: Federated Learning with Reinforcement Learning

This module implements federated RL with policy
aggregation across distributed agents.

Copyright 2026 Olaf Yunus Laitinen Imanov and Contributors
Released under EUPL 1.2
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import copy
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class FRLConfig:
    """Configuration for federated RL."""

    num_rounds: int = 50
    num_agents: int = 10
    agents_per_round: int = 5

    state_dim: int = 4
    action_dim: int = 2
    hidden_dim: int = 64

    learning_rate: float = 0.001
    gamma: float = 0.99
    episodes_per_round: int = 10

    seed: int = 42


class SimpleEnv:
    """Simple environment for RL."""

    def __init__(self, env_id: int = 0, seed: int = 0):
        self.env_id = env_id
        self.rng = np.random.RandomState(seed + env_id)

        self.state_dim = 4
        self.action_dim = 2

        self.state = None
        self.max_steps = 100
        self.steps = 0

    def reset(self) -> np.ndarray:
        self.state = self.rng.randn(self.state_dim).astype(np.float32)
        self.steps = 0
        return self.state

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        self.steps += 1

        # Simple dynamics: action affects reward
        reward = float(self.state[action] > 0)

        # State transition
        self.state = self.state + self.rng.randn(self.state_dim).astype(np.float32) * 0.1

        done = self.steps >= self.max_steps

        return self.state, reward, done, {}


class PolicyNetwork(nn.Module):
    """Policy network for RL."""

    def __init__(self, config: FRLConfig):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(config.state_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.action_dim)
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.net(state), dim=-1)

    def select_action(self, state: np.ndarray) -> Tuple[int, torch.Tensor]:
        state_t = torch.FloatTensor(state).unsqueeze(0)
        probs = self.forward(state_t)

        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action.item(), log_prob


class RLAgent:
    """Federated RL agent."""

    def __init__(
        self,
        agent_id: int,
        env: SimpleEnv,
        config: FRLConfig
    ):
        self.agent_id = agent_id
        self.env = env
        self.config = config

    def train(self, policy: nn.Module) -> Dict[str, Any]:
        """Train locally using REINFORCE."""
        local = copy.deepcopy(policy)
        optimizer = torch.optim.Adam(local.parameters(), lr=self.config.learning_rate)

        total_rewards = []

        for _ in range(self.config.episodes_per_round):
            log_probs = []
            rewards = []

            state = self.env.reset()
            done = False

            while not done:
                action, log_prob = local.select_action(state)
                next_state, reward, done, _ = self.env.step(action)

                log_probs.append(log_prob)
                rewards.append(reward)

                state = next_state

            total_rewards.append(sum(rewards))

            # Compute returns
            returns = []
            G = 0
            for r in reversed(rewards):
                G = r + self.config.gamma * G
                returns.insert(0, G)

            returns = torch.FloatTensor(returns)
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

            # Policy gradient update
            optimizer.zero_grad()

            policy_loss = 0
            for log_prob, G in zip(log_probs, returns):
                policy_loss += -log_prob * G

            policy_loss.backward()
            optimizer.step()

        return {
            "state_dict": {k: v.cpu() for k, v in local.state_dict().items()},
            "avg_reward": np.mean(total_rewards),
            "agent_id": self.agent_id
        }

    def evaluate(self, policy: nn.Module, num_episodes: int = 5) -> float:
        """Evaluate policy."""
        policy.eval()
        total_rewards = []

        for _ in range(num_episodes):
            state = self.env.reset()
            done = False
            episode_reward = 0

            while not done:
                with torch.no_grad():
                    action, _ = policy.select_action(state)
                state, reward, done, _ = self.env.step(action)
                episode_reward += reward

            total_rewards.append(episode_reward)

        return np.mean(total_rewards)


class FRLServer:
    """Federated RL server."""

    def __init__(
        self,
        policy: nn.Module,
        agents: List[RLAgent],
        config: FRLConfig
    ):
        self.policy = policy
        self.agents = agents
        self.config = config
        self.history: List[Dict] = []

    def aggregate(self, updates: List[Dict]) -> None:
        """Aggregate policy updates."""
        if not updates:
            return

        # Weight by reward
        rewards = [max(0.1, u["avg_reward"]) for u in updates]
        total_reward = sum(rewards)

        new_state = {}
        for key in updates[0]["state_dict"]:
            new_state[key] = sum(
                (r / total_reward) * u["state_dict"][key].float()
                for u, r in zip(updates, rewards)
            )

        self.policy.load_state_dict(new_state)

    def evaluate(self) -> float:
        """Evaluate global policy."""
        rewards = [
            agent.evaluate(self.policy, num_episodes=3)
            for agent in self.agents[:3]
        ]
        return np.mean(rewards)

    def train(self) -> List[Dict]:
        """Run federated RL training."""
        logger.info(f"Starting federated RL with {len(self.agents)} agents")

        for round_num in range(self.config.num_rounds):
            n = min(self.config.agents_per_round, len(self.agents))
            indices = np.random.choice(len(self.agents), n, replace=False)
            selected = [self.agents[i] for i in indices]

            updates = [agent.train(self.policy) for agent in selected]

            avg_train_reward = np.mean([u["avg_reward"] for u in updates])

            self.aggregate(updates)

            eval_reward = self.evaluate()

            record = {
                "round": round_num,
                "train_reward": avg_train_reward,
                "eval_reward": eval_reward
            }
            self.history.append(record)

            if (round_num + 1) % 10 == 0:
                logger.info(
                    f"Round {round_num + 1}: "
                    f"train={avg_train_reward:.2f}, eval={eval_reward:.2f}"
                )

        return self.history


def main():
    print("=" * 60)
    print("Tutorial 113: FL with RL")
    print("=" * 60)

    config = FRLConfig()
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    # Create agents with different environments
    agents = []
    for i in range(config.num_agents):
        env = SimpleEnv(env_id=i, seed=config.seed)
        agent = RLAgent(i, env, config)
        agents.append(agent)

    policy = PolicyNetwork(config)

    server = FRLServer(policy, agents, config)
    history = server.train()

    print("\n" + "=" * 60)
    print("Training Complete")
    print(f"Final Eval Reward: {history[-1]['eval_reward']:.2f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
```

---

## Key Insights

### Federated RL Challenges

1. **Environment heterogeneity**: Different dynamics
2. **Reward aggregation**: Weight by performance
3. **Exploration balance**: Diverse strategies
4. **Sample efficiency**: Limited episodes

---

## Exercises

1. **Exercise 1**: Add experience replay sharing
2. **Exercise 2**: Implement actor-critic FRL
3. **Exercise 3**: Design multi-task FRL
4. **Exercise 4**: Add curiosity-driven exploration

---

## References

1. Zhuo, H.H., et al. (2019). Federated reinforcement learning. In *ICML Workshop*.
2. Liu, B., et al. (2019). Lifelong federated reinforcement learning. *arXiv*.
3. Nadiger, C., et al. (2019). Federated reinforcement learning for fast personalization. In *AISTATS*.

---

*Copyright 2026 Olaf Yunus Laitinen Imanov and Contributors. Released under EUPL 1.2.*
