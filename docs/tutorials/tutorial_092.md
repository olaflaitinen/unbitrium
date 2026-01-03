# Tutorial 092: Personalized FL with pFedMe

This tutorial covers pFedMe (Personalized Federated Learning via Moreau Envelopes).

## Algorithm

$$
\min_{w} F(w) + \lambda \|w - \theta_k\|^2
$$

## Implementation Stub

```python
def pfedme_update(client_model, global_model, lr, lam, local_steps):
    # Personalization step
    theta = client_model.clone()
    for _ in range(local_steps):
        grad = compute_grad(theta)
        # Proximal update
        theta = theta - lr * (grad + lam * (theta - global_model))
    return theta
```

## Exercises

1. How does $\lambda$ control the personalization strength?
2. Compare pFedMe with pFedSim.
