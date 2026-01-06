# NMIRepresentations Validation Report

## Overview

Normalized Mutual Information (NMI) between representation clusters quantifies the alignment of learned features across clients. High NMI indicates consistent representations; low NMI suggests feature-space drift.

### Mathematical Formulation

$$
\text{NMI}(U, V) = \frac{I(U; V)}{\sqrt{H(U) \cdot H(V)}}
$$

where:
- $U$ and $V$ are cluster assignments from two representation spaces
- $I(U; V)$ is mutual information
- $H(U)$ and $H(V)$ are entropies

Mutual information:

$$
I(U; V) = \sum_{u, v} p(u, v) \log \frac{p(u, v)}{p(u) p(v)}
$$

### Implementation Reference

The implementation is located at `src/unbitrium/metrics/representation.py`.

---

## Invariants

### Invariant 1: Bounded Range

$$
0 \leq \text{NMI}(U, V) \leq 1
$$

**Verification**: All computed NMI values in $[0, 1]$.

### Invariant 2: Identity

$$
\text{NMI}(U, U) = 1
$$

**Verification**: Self-comparison yields 1.

### Invariant 3: Symmetry

$$
\text{NMI}(U, V) = \text{NMI}(V, U)
$$

**Verification**: Order-independent.

### Invariant 4: Independence

$$
U \perp V \implies \text{NMI}(U, V) = 0
$$

**Verification**: Independent clusterings yield zero (in expectation).

---

## Test Distributions

### Distribution 1: Identical Clusterings

**Input**: Same cluster assignments

**Expected Output**: NMI = 1.0

### Distribution 2: Random Clusterings

**Input**: Independent random assignments

**Expected Output**: NMI $\approx 0$ (close to zero)

### Distribution 3: Subset Alignment

**Input**: One clustering is refinement of another

**Expected Behavior**: NMI < 1 but positive

### Distribution 4: Permuted Clusters

**Input**: Same clusters but relabeled

**Expected Output**: NMI = 1.0 (label-invariant)

---

## Application in FL

### Client-Global Comparison

Compare client representations to global model:

1. Extract embeddings from global model
2. Extract embeddings from client model
3. Cluster each embedding space
4. Compute NMI between cluster assignments

### Expected Behavior by Heterogeneity

| Heterogeneity | Expected NMI |
|---------------|--------------|
| IID | 0.8 - 1.0 |
| Low non-IID | 0.6 - 0.8 |
| Moderate | 0.4 - 0.6 |
| High | 0.2 - 0.4 |
| Extreme | < 0.2 |

---

## Edge Cases

### Edge Case 1: Single Cluster

**Input**: All samples in one cluster

**Expected Behavior**:
- $H(U) = 0$
- NMI undefined (0/0)
- Return 0 or raise warning

### Edge Case 2: Perfect Correspondence

**Input**: One-to-one cluster mapping

**Expected Output**: NMI = 1.0

### Edge Case 3: Different Number of Clusters

**Input**: $|U| \neq |V|$

**Expected Behavior**:
- Still computable
- NMI measures overlap regardless of cluster count

---

## Reproducibility

### Usage Example

```python
from unbitrium.metrics import NMIRepresentations
from sklearn.cluster import KMeans

metric = NMIRepresentations(n_clusters=10)

# Get representations
global_reps = global_model.encode(data)
client_reps = client_model.encode(data)

# Cluster and compute NMI
nmi = metric.compute(global_reps, client_reps)
print(f"NMI: {nmi:.4f}")
```

### Clustering Configuration

```python
metric = NMIRepresentations(
    n_clusters=10,
    clustering_method="kmeans",
    random_state=42,
)
```

---

## Security Considerations

### Information Content

NMI reveals:
- Degree of representation alignment
- Feature-space structure similarity

### Mitigations

1. Compute on public validation data only
2. Report aggregate statistics

---

## Complexity Analysis

### Time Complexity

Clustering: $O(N \cdot K \cdot d \cdot I)$

NMI computation: $O(N \cdot C_U \cdot C_V)$

Total: Dominated by clustering.

### Space Complexity

$$
S = O(N \cdot d + N)
$$

---

## Alternative Metrics

### Adjusted Mutual Information

Corrected for chance:

$$
\text{AMI}(U, V) = \frac{I(U; V) - \mathbb{E}[I]}{\max(H(U), H(V)) - \mathbb{E}[I]}
$$

### Centered Kernel Alignment

Kernel-based similarity without clustering:

$$
\text{CKA}(X, Y) = \frac{\text{HSIC}(X, Y)}{\sqrt{\text{HSIC}(X, X) \cdot \text{HSIC}(Y, Y)}}
$$

---

## References

1. Vinh, N. X., Epps, J., & Bailey, J. (2010). Information theoretic measures for clusterings comparison: Variants, properties, normalization and correction for chance. *JMLR*, 11, 2837-2854.

2. Kornblith, S., et al. (2019). Similarity of neural network representations revisited. In *ICML*.

3. Nguyen, T., et al. (2020). Wide neural networks of any depth evolve as linear models under gradient descent. In *JMLR*.

---

## Changelog

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2026-01-04 | Initial validation report |

---

Copyright 2026 Olaf Yunus Laitinen Imanov and Contributors. Released under EUPL 1.2.
