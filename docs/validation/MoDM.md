# MoDM Validation Report

## Overview

Mixture-of-Dirichlet-Multinomials (MoDM) extends standard Dirichlet partitioning by modeling client distributions as a mixture of Dirichlet components. This enables multimodal non-IID structures where clients cluster into distinct distribution patterns.

### Mathematical Formulation

Each client is assigned to a mixture component:

$$
z_k \sim \text{Categorical}(\pi)
$$

where $\pi = (\pi_1, \ldots, \pi_M)$ are mixture weights.

Given component assignment $z_k = m$, sample label proportions:

$$
p_k | z_k = m \sim \text{Dirichlet}(\alpha_m)
$$

where $\alpha_m \in \mathbb{R}^C_{>0}$ is the concentration vector for component $m$.

### Implementation Reference

The implementation is located at `src/unbitrium/partitioning/modm.py`.

---

## Invariants

### Invariant 1: Mixture Weights Normalization

Mixture weights form a probability distribution:

$$
\sum_{m=1}^M \pi_m = 1, \quad \pi_m \geq 0
$$

**Verification**: Weights validated to sum to unity.

### Invariant 2: Component Assignment Consistency

Each client assigned to exactly one component:

$$
z_k \in \{1, \ldots, M\} \text{ for all } k
$$

**Verification**: Assignment is deterministic given seed.

### Invariant 3: Reduction to Dirichlet

With $M = 1$, reduces to standard Dirichlet:

$$
M = 1 \implies \text{MoDM} \equiv \text{Dirichlet}(\alpha_1)
$$

**Verification**: Single component produces Dirichlet results.

### Invariant 4: Reproducibility

Fixed seed produces identical partitions.

---

## Test Distributions

### Distribution 1: Bimodal Label Skew

**Configuration**:
- Mixture components: $M = 2$
- $\pi = (0.5, 0.5)$
- $\alpha_1 = (10, 10, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1)$ (classes 0-1 dominant)
- $\alpha_2 = (0.1, 0.1, 10, 10, 10, 0.1, 0.1, 0.1, 0.1, 0.1)$ (classes 2-4 dominant)

**Expected Behavior**:
- Clients split into two groups
- Group 1: primarily classes 0-1
- Group 2: primarily classes 2-4
- Clear cluster structure in label space

### Distribution 2: Multimodal with Varying Sizes

**Configuration**:
- $M = 4$
- $\pi = (0.1, 0.2, 0.3, 0.4)$
- Distinct $\alpha_m$ for each component

**Expected Behavior**:
- Four client groups with different sizes
- Largest group (40%) most influential
- Heterogeneity within and between groups

### Distribution 3: Hierarchical Structure

**Configuration**:
- $M = 5$ (representing domains: natural, urban, medical, sketch, satellite)
- Each component has domain-specific class preferences

**Expected Behavior**:
- Simulates cross-domain federated learning
- Each domain has characteristic label distribution

---

## Expected Behavior

### Component Configuration Guide

| Scenario | $M$ | $\pi$ Distribution | $\alpha$ Pattern |
|----------|-----|-------------------|------------------|
| Bimodal | 2 | Uniform | Complementary |
| Geographic | 3-5 | By region size | Regional priors |
| Device type | 3 | By market share | Usage patterns |
| Temporal | 2-4 | By time period | Trend shifts |

### Metric Ranges

| Metric | Range | Notes |
|--------|-------|-------|
| `num_components` | $[1, K]$ | Number of mixture components |
| `component_sizes` | Varies | Clients per component |
| `inter_component_emd` | $[0, 1]$ | Distance between components |
| `intra_component_variance` | $[0, \infty)$ | Variance within component |

---

## Edge Cases

### Edge Case 1: Single Component

**Input**: $M = 1$

**Expected Behavior**:
- Equivalent to Dirichlet partitioner
- All clients from same distribution

### Edge Case 2: Equal Components

**Input**: $M = K$ (one component per client)

**Expected Behavior**:
- Each client has unique concentration
- Maximum between-client heterogeneity

### Edge Case 3: Empty Component

**Input**: $\pi_m = 0$ for some $m$

**Expected Behavior**:
- Component $m$ receives no clients
- Effective $M' = M - 1$

### Edge Case 4: Degenerate Concentration

**Input**: $\alpha_m = (10^6, 0, \ldots, 0)$

**Expected Behavior**:
- Component $m$ clients have only class 0
- Extreme specialization

---

## Reproducibility

### Seed Configuration

```python
from unbitrium.partitioning import MoDM

partitioner = MoDM(
    num_components=3,
    mixture_weights=[0.3, 0.4, 0.3],
    component_alphas=[
        [1.0] * 10,  # Uniform
        [0.1, 0.1, 10.0, 10.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
        [10.0, 10.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
    ],
    num_clients=100,
    seed=42,
)
```

---

## Security Considerations

### Information Leakage

Mixture structure may reveal:
- Cluster membership of clients
- Domain or group affiliations

### Mitigations

1. Differential privacy on component assignments
2. Shuffle clients before revealing to server

---

## Complexity Analysis

### Time Complexity

$$
T = O(K + N)
$$

**Breakdown**:
- Component assignment: $O(K)$
- Sample assignment: $O(N)$

### Space Complexity

$$
S = O(M \cdot C + K \cdot C + N)
$$

---

## References

1. Yurochkin, M., et al. (2019). Bayesian nonparametric federated learning of neural networks. In *ICML*.

2. Marfoq, O., et al. (2021). Federated multi-task learning under a mixture of distributions. In *NeurIPS*.

3. Ghosh, A., et al. (2020). An efficient framework for clustered federated learning. In *NeurIPS*.

---

## Changelog

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2026-01-04 | Initial validation report |

---

Copyright 2026 Olaf Yunus Laitinen Imanov and Contributors. Released under EUPL 1.2.
