# FedDyn Validation Report

## Overview

Federated Dynamic Regularization (FedDyn) is an advanced aggregation algorithm that uses dynamic regularization to achieve linear convergence rates under heterogeneous data distributions. It maintains per-client state vectors that adapt the regularization to each client's deviation from the global optimum.

### Mathematical Formulation

Each client $k$ minimizes a dynamically regularized objective:

$$
\min_{w} F_k(w) - \langle h_k^t, w \rangle + \frac{\alpha}{2}\|w\|^2
$$

where:
- $F_k(w)$ is the local empirical loss
- $h_k^t$ is a client-specific state vector updated each round
- $\alpha > 0$ is the regularization coefficient

The state vector is updated as:

$$
h_k^{t+1} = h_k^t - \alpha(w_k^{t+1} - w^t)
$$

The global aggregation uses a corrected average:

$$
w^{t+1} = \frac{1}{K}\sum_{k=1}^K w_k^{t+1} - \frac{1}{\alpha}\bar{h}^{t+1}
$$

where $\bar{h}^{t+1} = \frac{1}{K}\sum_k h_k^{t+1}$.

### Implementation Reference

The implementation is located at `src/unbitrium/aggregators/feddyn.py`.

---

## Invariants

### Invariant 1: State Conservation

The sum of client state vectors tracks cumulative drift:

$$
\sum_k h_k^t = -\alpha \sum_{\tau=0}^{t-1} \sum_k (w_k^{\tau+1} - w^\tau)
$$

**Verification**: Property-based tests confirm state vector sum equals accumulated drift.

### Invariant 2: Linear Convergence

Under strong convexity, FedDyn achieves linear convergence:

$$
\mathbb{E}[\|w^t - w^*\|^2] \leq (1 - \mu/(\mu + \alpha))^t \|w^0 - w^*\|^2
$$

**Verification**: Convergence curves on strongly convex objectives match theoretical rate.

### Invariant 3: Stateful Consistency

Client states must persist across rounds:

$$
h_k^{t+1} = f(h_k^t, w_k^{t+1}, w^t)
$$

**Verification**: State vectors are correctly maintained in aggregator state.

### Invariant 4: Determinism

Identical inputs produce identical outputs.

---

## Test Distributions

### Distribution 1: Strongly Convex Objective

**Configuration**:
- Objective: Ridge regression with $\lambda = 0.01$
- Clients: $K = 10$
- $\alpha = 0.1$

**Expected Behavior**:
- Linear convergence to optimal solution
- Final loss within $10^{-6}$ of optimal

### Distribution 2: Non-IID Label Skew

**Configuration**:
- Dataset: CIFAR-10
- Partitioning: Dirichlet $\alpha_{dir} = 0.1$
- $\alpha = 0.01$

**Expected Behavior**:
- Outperforms FedAvg by 8-12% accuracy
- Stable convergence despite heterogeneity

### Distribution 3: Partial Participation

**Configuration**:
- Clients: $K = 100$
- Participation rate: 10%
- $\alpha = 0.01$

**Expected Behavior**:
- State vectors maintained for non-participating clients
- Convergence slower but stable

---

## Expected Behavior

### Hyperparameter Sensitivity

| $\alpha$ Range | Effect | Recommendation |
|----------------|--------|----------------|
| $[0.001, 0.01)$ | Weak regularization | Large datasets |
| $[0.01, 0.1)$ | Moderate regularization | Default choice |
| $[0.1, 1.0)$ | Strong regularization | Extreme non-IID |

### Metric Ranges

| Metric | Range | Notes |
|--------|-------|-------|
| `state_norm` | $[0, \infty)$ | Norm of aggregated state vectors |
| `correction_magnitude` | $[0, \infty)$ | Size of dynamic correction |
| `alpha` | $(0, \infty)$ | Regularization coefficient |

---

## Edge Cases

### Edge Case 1: First Round (No Prior State)

**Input**: Round $t = 0$

**Expected Behavior**:
- Initialize $h_k^0 = 0$ for all clients
- Equivalent to standard aggregation with L2 regularization

### Edge Case 2: New Client Joins

**Input**: Client $k$ participates for first time at round $t > 0$

**Expected Behavior**:
- Initialize $h_k^t = 0$
- Client state builds up over subsequent rounds

### Edge Case 3: Client Drops Out

**Input**: Client $k$ stops participating at round $t$

**Expected Behavior**:
- State $h_k^t$ is frozen
- May cause drift in aggregation if prolonged

---

## Reproducibility

### Seed Configuration

```python
def set_seed(seed: int = 42) -> None:
    import random, numpy as np, torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
```

### Validation Environment

| Component | Version |
|-----------|---------|
| Python | 3.12.0 |
| PyTorch | 2.4.0 |
| Unbitrium | 1.0.0 |

---

## Security Considerations

### State Vector Privacy

Client state vectors $h_k^t$ encode cumulative training history:
- May reveal information about local data distribution shifts
- Should be protected similarly to model updates

### Mitigations

1. Apply differential privacy to state updates
2. Use secure aggregation for state vector transmission
3. Periodic state reset to limit temporal leakage

---

## Complexity Analysis

### Time Complexity

Per-round: $O(K \cdot P)$ where $P$ is parameter count

### Space Complexity

$$
S(K, P) = O(K \cdot P)
$$

**Note**: FedDyn requires storing per-client state vectors, increasing memory by factor $K$ compared to FedAvg.

---

## References

1. Acar, D. A. E., Zhao, Y., Navarro, R. M., Mattina, M., Whatmough, P. N., & Saber, V. (2021). Federated learning based on dynamic regularization. In *ICLR*.

2. Li, T., et al. (2020). Federated optimization in heterogeneous networks. In *MLSys*.

---

## Changelog

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2026-01-04 | Initial validation report |

---

Copyright 2026 Olaf Yunus Laitinen Imanov and Contributors. Released under EUPL 1.2.
