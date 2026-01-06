# ClientDriftNorm Validation Report

## Overview

Client Drift Norm measures the divergence between local client models and the global model in parameter space. It quantifies how far clients deviate during local training.

### Mathematical Formulation

L2 norm of parameter difference:

$$
\text{Drift}_k^t = \|w_k^t - w^t\|_2
$$

where:
- $w_k^t$ is client $k$'s local model at round $t$
- $w^t$ is the global model at round $t$

Relative drift:

$$
\text{RelDrift}_k^t = \frac{\|w_k^t - w^t\|_2}{\|w^t\|_2}
$$

### Implementation Reference

The implementation is located at `src/unbitrium/metrics/optimization.py`.

---

## Invariants

### Invariant 1: Non-negativity

$$
\text{Drift}_k^t \geq 0
$$

**Verification**: Norm is always non-negative.

### Invariant 2: Zero for Identical Models

$$
w_k = w \implies \text{Drift}_k = 0
$$

**Verification**: Same model yields zero drift.

### Invariant 3: Triangle Inequality

$$
\|w_k - w_j\| \leq \|w_k - w\| + \|w - w_j\|
$$

**Verification**: L2 norm is a proper metric.

### Invariant 4: Growth with Local Epochs

Drift typically increases with local training:

$$
E_1 < E_2 \implies \mathbb{E}[\text{Drift}|E_1] \leq \mathbb{E}[\text{Drift}|E_2]
$$

**Verification**: More local epochs lead to higher drift (on average).

---

## Test Distributions

### Distribution 1: Zero Local Steps

**Configuration**:
- Local epochs $E = 0$ (client returns global model)

**Expected Output**: Drift = 0 for all clients

### Distribution 2: Single Local Step

**Configuration**:
- $E = 1$, batch size = 32
- Various non-IID levels

**Expected Behavior**:
- Drift proportional to learning rate$\times$ gradient norm
- Non-IID increases drift

### Distribution 3: Many Local Steps

**Configuration**:
- $E = 10$
- Dirichlet $\alpha = 0.1$

**Expected Behavior**:
- High drift values
- Significant client divergence
- FedProx naturally reduces this

### Distribution 4: Over Rounds

**Configuration**:
- Track drift over $T = 100$ rounds

**Expected Behavior**:
- Drift may stabilize as model converges
- Or oscillate in non-IID settings

---

## Expected Behavior

### Drift Interpretation

| Relative Drift | Severity | Recommended Action |
|----------------|----------|-------------------|
| 0.0 - 0.01 | Minimal | FedAvg works |
| 0.01 - 0.1 | Low | Monitor |
| 0.1 - 0.5 | Moderate | Consider FedProx |
| 0.5 - 1.0 | High | Use regularization |
| 1.0+ | Severe | Reduce local epochs or use SCAFFOLD |

### Factors Affecting Drift

| Factor | Effect on Drift |
|--------|-----------------|
| Local epochs $E$ | Increases with $E$ |
| Learning rate $\eta$ | Increases with $\eta$ |
| Data heterogeneity | Increases with non-IID |
| Model complexity | Generally increases |
| Batch size | Inversely related |

---

## Edge Cases

### Edge Case 1: Zero Global Model

**Input**: $w^t = 0$ (e.g., at initialization)

**Expected Behavior**:
- Relative drift undefined (0/0)
- Use absolute drift or handle specially

### Edge Case 2: Converged Model

**Input**: Near convergence

**Expected Behavior**:
- Gradients small
- Drift approaches zero

### Edge Case 3: Exploding Drift

**Input**: Learning rate too high

**Expected Behavior**:
- Drift explodes
- Training diverges
- Should trigger early stopping

---

## Reproducibility

### Usage Example

```python
from unbitrium.metrics import ClientDriftNorm

metric = ClientDriftNorm(normalize=True)

# After local training
global_params = global_model.state_dict()
client_params = client_model.state_dict()

drift = metric.compute(client_params, global_params)
print(f"Client Drift: {drift:.4f}")
```

### Aggregate Statistics

```python
# Compute for all clients
drifts = []
for client in clients:
    drift = metric.compute(client.model, global_model)
    drifts.append(drift)

print(f"Mean Drift: {np.mean(drifts):.4f}")
print(f"Max Drift: {np.max(drifts):.4f}")
print(f"Std Drift: {np.std(drifts):.4f}")
```

---

## Security Considerations

### Information Content

Drift reveals:
- Client update magnitude
- Relative training dynamics

### Mitigations

1. Report aggregate drift only
2. Normalize across rounds

---

## Complexity Analysis

### Time Complexity

$$
T = O(P)
$$

Single pass over parameters.

### Space Complexity

$$
S = O(P)
$$

Store both model copies.

---

## Relationship to FedProx

FedProx regularization explicitly penalizes drift:

$$
\min_w F_k(w) + \frac{\mu}{2}\|w - w^t\|^2
$$

The drift norm is exactly the term being penalized.

### Optimal Mu Selection

Empirical heuristic:

$$
\mu_{opt} \approx \frac{1}{\mathbb{E}[\text{Drift}_k^2]}
$$

Scale $\mu$ inversely with typical drift.

---

## References

1. Li, T., et al. (2020). Federated optimization in heterogeneous networks. In *MLSys*.

2. Karimireddy, S. P., et al. (2020). SCAFFOLD: Stochastic controlled averaging for federated learning. In *ICML*.

3. Wang, S., et al. (2020). Tackling the objective inconsistency problem in heterogeneous federated optimization. In *NeurIPS*.

---

## Changelog

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2026-01-04 | Initial validation report |

---

Copyright 2026 Olaf Yunus Laitinen Imanov and Contributors. Released under EUPL 1.2.
