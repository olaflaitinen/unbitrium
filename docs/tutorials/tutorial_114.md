# Tutorial 114: Federated Meta-Learning

This tutorial covers MAML-style meta-learning in FL.

## Approach

- Learn initialization that generalizes across clients (tasks).
- Each client performs few-step adaptation.

$$
\theta^* = \arg\min_\theta \sum_k \mathcal{L}_k(\theta - \alpha \nabla \mathcal{L}_k(\theta))
$$

## Configuration

```yaml
meta:
  method: "maml"
  inner_lr: 0.01
  inner_steps: 5
```

## Exercises

1. Second-order gradients and communication overhead.
2. Reptile as a first-order approximation.
