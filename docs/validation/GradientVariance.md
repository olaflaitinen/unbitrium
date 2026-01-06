# GradientVariance Validation Report

## Overview

Gradient Variance measures the dispersion of client gradients around the mean gradient. High variance indicates significant client drift, a key challenge in heterogeneous federated learning.

### Mathematical Formulation

$$
\sigma^2 = \frac{1}{K} \sum_{k=1}^{K} \|\nabla F_k(w) - \bar{\nabla}F(w)\|^2
$$

where:
- $\nabla F_k(w)$ is the gradient on client $k$
- $\bar{\nabla}F(w) = \frac{1}{K}\sum_k \nabla F_k(w)$ is the mean gradient
- $K$ is the number of clients

Weighted variant:

$$
\sigma^2_w = \sum_{k=1}^{K} \frac{n_k}{N} \|\nabla F_k(w) - \bar{\nabla}F(w)\|^2
$$

### Implementation Reference

The implementation is located at `src/unbitrium/metrics/optimization.py`.

---

## Invariants

### Invariant 1: Non-negativity

$$
\sigma^2 \geq 0
$$

**Verification**: Variance is always non-negative.

### Invariant 2: Zero for Identical Gradients

$$
\nabla F_k = \nabla F_j \text{ for all } k, j \implies \sigma^2 = 0
$$

**Verification**: Identical gradients yield zero variance.

### Invariant 3: Scale Invariance (Normalized)

For normalized variant:

$$
\sigma^2_{norm} = \frac{\sigma^2}{\|\bar{\nabla}F\|^2}
$$

**Verification**: Normalized variance independent of gradient magnitude.

### Invariant 4: Additivity Over Dimensions

$$
\sigma^2 = \sum_{i=1}^{P} \sigma^2_i
$$

where $\sigma^2_i$ is variance in dimension $i$.

---

## Test Distributions

### Distribution 1: IID Data

**Configuration**:
- Clients: $K = 10$
- IID data split
- Same model and batch size

**Expected Behavior**:
- Low variance ($\sigma^2 < 0.1 \|\bar{\nabla}\|^2$)
- Gradients point in similar directions

### Distribution 2: High Non-IID (Dirichlet 0.1)

**Configuration**:
- Clients: $K = 20$
- Dirichlet $\alpha = 0.1$

**Expected Behavior**:
- High variance ($\sigma^2 > \|\bar{\nabla}\|^2$)
- Gradients point in diverse directions
- Cosine similarity across clients: 0.3-0.6

### Distribution 3: Single Epoch vs Multiple

**Configuration**:
- Compare $E = 1$ vs $E = 5$ local epochs

**Expected Behavior**:
- Higher $E$ leads to higher variance (more drift)
- Variance increases nonlinearly with $E$

### Distribution 4: Byzantine Client

**Configuration**:
- One client with inverted gradients

**Expected Behavior**:
- Variance dramatically elevated
- Outlier detection possible via variance

---

## Expected Behavior

### Variance Interpretation

| $\sigma^2 / \|\bar{\nabla}\|^2$ | Heterogeneity | Recommended Action |
|--------------------------------|---------------|-------------------|
| 0.0 - 0.1 | Minimal | Standard FedAvg |
| 0.1 - 0.5 | Low | FedAvg works |
| 0.5 - 1.0 | Moderate | Consider FedProx |
| 1.0 - 3.0 | High | Use SCAFFOLD/FedDyn |
| 3.0+ | Severe | Advanced methods needed |

### Correlation with Convergence

FedProx regularization threshold:
- $\sigma^2 > 0.3$: Proximal term beneficial
- Paper finding: Accuracy improves with $\mu$ tuned to variance

---

## Edge Cases

### Edge Case 1: Single Client

**Input**: $K = 1$

**Expected Behavior**:
- Variance = 0 (no other clients to compare)
- Undefined for meaningful measurement

### Edge Case 2: Zero Mean Gradient

**Input**: $\bar{\nabla}F = 0$

**Expected Behavior**:
- Normalized variance undefined
- Use unnormalized variance

### Edge Case 3: High-Dimensional Gradients

**Input**: $P = 10^9$ parameters

**Expected Behavior**:
- Computation feasible (element-wise)
- Memory for storing gradients is bottleneck

---

## Reproducibility

### Usage Example

```python
from unbitrium.metrics import GradientVariance

metric = GradientVariance(normalize=True)

# Collect gradients from all clients
gradients = [client.compute_gradient(global_model) for client in clients]

variance = metric.compute(gradients)
print(f"Gradient Variance: {variance:.4f}")
```

### Per-Layer Analysis

```python
# Analyze variance per layer
layer_variances = metric.compute_per_layer(gradients, model)
for layer, var in layer_variances.items():
    print(f"{layer}: {var:.4f}")
```

---

## Security Considerations

### Information Content

Gradient variance reveals:
- Data distribution heterogeneity
- Model update disagreement

### Mitigations

1. Compute variance on aggregated data only
2. Differential privacy on gradient statistics

---

## Complexity Analysis

### Time Complexity

$$
T = O(K \cdot P)
$$

Two passes: compute mean, then compute variance.

### Space Complexity

$$
S = O(K \cdot P)
$$

Must store all client gradients.

---

## Related Metrics

### Gradient Dissimilarity

$$
\delta = \max_k \|\nabla F_k - \bar{\nabla}F\|
$$

Maximum rather than average deviation.

### Cosine Disagreement

$$
\text{CD} = \frac{1}{K}\sum_k \left(1 - \frac{\langle \nabla F_k, \bar{\nabla}F \rangle}{\|\nabla F_k\| \|\bar{\nabla}F\|}\right)
$$

Directional disagreement.

---

## References

1. Li, T., et al. (2020). Federated optimization in heterogeneous networks. In *MLSys*.

2. Karimireddy, S. P., et al. (2020). SCAFFOLD: Stochastic controlled averaging for federated learning. In *ICML*.

3. Woodworth, B., et al. (2020). Is local SGD better than minibatch SGD? In *ICML*.

---

## Changelog

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2026-01-04 | Initial validation report |

---

Copyright 2026 Olaf Yunus Laitinen Imanov and Contributors. Released under EUPL 1.2.
