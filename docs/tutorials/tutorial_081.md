# Tutorial 081: Advanced Privacy Accounting with RDP

This tutorial demonstrates how to implement and track Renyi Differential Privacy (RDP) budgets.

## Background and Notation

- $\alpha$: Order of Renyi divergence.
- $\epsilon(\delta)$: Privacy budget at failure probability $\delta$.
- **Composition**: Properties of RDP allow tighter composition analysis than standard $(\epsilon, \delta)$-DP.

## Configuration File (YAML)

```yaml
experiment:
  name: "rdp_accounting_demo"

privacy:
  mechanism: "gaussian"
  noise_multiplier: 1.0
  max_grad_norm: 1.0
  orders: [2, 4, 8, 16, 32]
  target_delta: 1e-5

training:
  rounds: 50
  batch_size: 256
  sample_rate: 0.01  # q = L/N
```

## Minimal Runnable Code Example

```python
import numpy as np
from typing import List

def compute_rdp(q: float, sigma: float, steps: int, orders: List[float]) -> List[float]:
    """
    Computes RDP for Gaussian mechanism with subsampling.
    References Mironov et al.
    """
    rdp = []
    for alpha in orders:
        # Approximate formula for subsampled Gaussian
        eps = 3.5 * steps * (q**2) / (sigma**2) * alpha # Simplified
        rdp.append(eps)
    return rdp

def convert_to_approx_dp(rdp: List[float], orders: List[float], delta: float) -> float:
    min_epsilon = float('inf')
    for eps_alpha, alpha in zip(rdp, orders):
        epsilon = eps_alpha + np.log(1/delta) / (alpha - 1)
        min_epsilon = min(min_epsilon, epsilon)
    return min_epsilon

# Simulation
q = 0.01
sigma = 1.0
steps = 500
orders = [2, 4, 8, 32]

rdp_budget = compute_rdp(q, sigma, steps, orders)
final_epsilon = convert_to_approx_dp(rdp_budget, orders, 1e-5)

print(f"Final Privacy Cost: epsilon={final_epsilon:.4f} at delta=1e-5")
```

## Expected Outputs

- Step-by-step privacy cost accumulation.
- Final epsilon value.
- Warning if budget exceeds pre-defined threshold.

## RDP Composition

```mermaid
graph LR
    A[Round 1] -->|RDP(alpha)| B[Accumulator]
    C[Round 2] -->|RDP(alpha)| B
    D[Round T] -->|RDP(alpha)| B
    B -->|Convert| E[Final (eps, delta)]
```

## Exercises

1. How does $q$ (sampling rate) affect the privacy cost?
2. Why is RDP better for composition than strong composition theorems?
3. Implement the exact analytical tracking for subsampled Gaussian.
