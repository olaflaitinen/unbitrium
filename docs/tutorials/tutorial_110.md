# Tutorial 110: Federated Reinforcement Learning

This tutorial covers FL for RL: multiple agents learning a shared policy.

## Setting

- Agents interact with different environments.
- Goal: Learn a generalizable policy.

## Configuration

```yaml
algorithm: "federated_ppo"
environment: "CartPole-v1"

training:
  rollout_length: 128
  gamma: 0.99
```

## Policy Flow

```mermaid
sequenceDiagram
    Agent1->>Server: Send Policy Update
    Agent2->>Server: Send Policy Update
    Server->>Server: Aggregate Policies
    Server->>Agent1: Broadcast Global Policy
    Server->>Agent2: Broadcast Global Policy
```

## Exercises

1. Variance in updates due to environment stochasticity.
2. Asynchronous vs synchronous federated RL.
