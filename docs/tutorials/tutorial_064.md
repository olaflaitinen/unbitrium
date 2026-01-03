# Tutorial 064: Model Poisoning Attack

Model Poisoning (or Gradient Poisoning) involves malicious clients sending crafted updates to corrupt the global model or insert backdoors.

## Attack Implementation

A simple attack is **Sign Flipping** or **Random Noise**, or specifically crafting a vector to move the global model in an adverse direction.

```python
import torch

def malicious_update_sign_flip(client_model, scale=1.0):
    """
    Multiplies weights by -scale (Sign Flipping).
    """
    poisoned_state = {}
    for k, v in client_model.state_dict().items():
        if isinstance(v, torch.Tensor):
            poisoned_state[k] = v * -scale
        else:
            poisoned_state[k] = v
    return poisoned_state

def malicious_update_noise(client_model, noise_std=2.0):
    """
    Adds large Gaussian noise.
    """
    poisoned_state = {}
    for k, v in client_model.state_dict().items():
        if isinstance(v, torch.Tensor):
            noise = torch.randn_like(v) * noise_std
            poisoned_state[k] = v + noise
        else:
            poisoned_state[k] = v
    return poisoned_state
```

## Integrating into Simulation

In `FederatedSimulator`:
1.  Define a set of Malicious Client IDs.
2.  Override their `train()` method or intercept their update before aggregation.
3.  Inject the poisoned update.

```python
# Custom Malicious Client
class MaliciousClient(Client):
    def train(self, **kwargs):
        # Perform normal training to get a plausible update magnitude?
        # Or just return static poison?

        # 1. Normal Train
        res = super().train(**kwargs)

        # 2. Poison
        real_sd = self.model.state_dict()
        poisoned_sd = {k: v * -5.0 for k,v in real_sd.items()} # Explicit attack

        res["state_dict"] = poisoned_sd
        return res
```

Use robust aggregators like `Krum` or `TrimmedMean` to defend against these.
