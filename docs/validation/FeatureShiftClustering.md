# FeatureShiftClustering Validation Report

## Overview

Feature Shift Clustering partitions data based on feature-space clustering, assigning samples to clients according to their cluster membership. This simulates covariate shift where clients have distinct feature distributions.

### Mathematical Formulation

Given feature representations $\phi(x)$ for samples $x$, perform clustering:

$$
c(x) = \arg\min_{j \in \{1, \ldots, K\}} \|\phi(x) - \mu_j\|^2
$$

where $\mu_j$ is the centroid of cluster $j$ (corresponding to client $j$).

Assignment rule:

$$
\mathcal{D}_k = \{(x, y) : c(x) = k\}
$$

### Implementation Reference

The implementation is located at `src/unbitrium/partitioning/feature_shift.py`.

---

## Invariants

### Invariant 1: Complete Assignment

Every sample assigned to exactly one client:

$$
\bigcup_{k=1}^K \mathcal{D}_k = \mathcal{D}, \quad \mathcal{D}_i \cap \mathcal{D}_j = \emptyset \text{ for } i \neq j
$$

**Verification**: No samples lost or duplicated.

### Invariant 2: Cluster Consistency

Samples in same cluster share feature similarity:

$$
x_1, x_2 \in \mathcal{D}_k \implies \|\phi(x_1) - \phi(x_2)\| \leq R_k
$$

where $R_k$ is cluster $k$'s radius.

**Verification**: Intra-cluster distances bounded.

### Invariant 3: Between-Cluster Separation

Clusters are reasonably separated:

$$
\min_{k \neq j} \|\mu_k - \mu_j\| > 0
$$

**Verification**: Distinct centroids for each client.

### Invariant 4: Reproducibility

Fixed seed produces identical clusters.

---

## Test Distributions

### Distribution 1: MNIST Digit Styles

**Configuration**:
- Dataset: MNIST
- Feature extractor: CNN embeddings
- Clients: $K = 10$

**Expected Behavior**:
- Clients receive visually similar digits
- Writing style variation across clients
- Labels may be mixed within clients

### Distribution 2: CIFAR-10 Visual Similarity

**Configuration**:
- Dataset: CIFAR-10
- Feature extractor: ResNet-18 (pretrained)
- Clients: $K = 20$

**Expected Behavior**:
- Clients specialized by visual appearance
- Animals grouped separately from vehicles
- Natural feature-based clustering

### Distribution 3: Domain Adaptation Setup

**Configuration**:
- Dataset: Multi-domain (photos, sketches, paintings)
- Feature extractor: Domain-invariant network
- $K = 3$ (one per domain)

**Expected Behavior**:
- Near-perfect domain separation
- Each client has single domain
- Feature shift maximized between clients

### Distribution 4: Overlapping Clusters

**Configuration**:
- Use soft clustering (GMM)
- Allow probabilistic assignment

**Expected Behavior**:
- Some samples near cluster boundaries
- Smooth transition between clients

---

## Expected Behavior

### Clustering Algorithm Options

| Algorithm | Characteristics | Best For |
|-----------|-----------------|----------|
| K-Means | Hard, spherical clusters | Balanced clients |
| K-Means++ | Better initialization | General use |
| GMM | Soft, elliptical | Overlapping |
| Hierarchical | Dendrogram structure | Unknown $K$ |
| DBSCAN | Density-based | Variable sizes |

### Metric Ranges

| Metric | Range | Notes |
|--------|-------|-------|
| `silhouette_score` | $[-1, 1]$ | Cluster quality |
| `inertia` | $[0, \infty)$ | Within-cluster variance |
| `cluster_sizes` | Varies | Samples per client |
| `feature_dimension` | $(0, \infty)$ | Embedding dimension |

---

## Edge Cases

### Edge Case 1: More Clients Than Natural Clusters

**Input**: Data has 3 natural clusters, $K = 10$ requested

**Expected Behavior**:
- Clusters subdivided
- Some clients very similar
- Reduced between-cluster variance

### Edge Case 2: Single Natural Cluster

**Input**: Homogeneous data, $K = 10$

**Expected Behavior**:
- Artificial subdivision
- Clients similar in features
- Quantity-based differentiation

### Edge Case 3: High-Dimensional Features

**Input**: $d = 10000$ feature dimensions

**Expected Behavior**:
- Curse of dimensionality effects
- Consider PCA preprocessing
- Cosine distance may outperform Euclidean

### Edge Case 4: Class Imbalance

**Input**: Dataset with 90% class A, 10% class B

**Expected Behavior**:
- Class B samples may concentrate in few clients
- Label skew emerges from feature clustering

---

## Reproducibility

### Seed Configuration

```python
from unbitrium.partitioning import FeatureShiftClustering

partitioner = FeatureShiftClustering(
    num_clients=10,
    clustering_algorithm="kmeans++",
    feature_extractor="resnet18",
    seed=42,
)
```

### Feature Extraction

```python
# Precompute features for reproducibility
features = partitioner.extract_features(dataset)
partitions = partitioner.cluster_and_assign(features)
```

---

## Security Considerations

### Feature Leakage

Clustering reveals:
- Natural data structure
- Similarity relationships
- Potential domain information

### Mitigations

1. Use privacy-preserving clustering
2. Add noise to feature representations
3. Shuffle cluster-client mapping

---

## Complexity Analysis

### Time Complexity

Feature extraction:
$$
T_{extract} = O(N \cdot T_{model})
$$

Clustering:
$$
T_{cluster} = O(N \cdot K \cdot d \cdot I)
$$

where $I$ is number of iterations.

### Space Complexity

$$
S = O(N \cdot d + K \cdot d)
$$

---

## References

1. Koh, P. W., et al. (2021). WILDS: A benchmark of in-the-wild distribution shifts. In *ICML*.

2. Gulrajani, I., & Lopez-Paz, D. (2021). In search of lost domain generalization. In *ICLR*.

3. Luo, M., et al. (2021). No fear of heterogeneity: Classifier calibration for federated learning with non-IID data. In *NeurIPS*.

---

## Changelog

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2026-01-04 | Initial validation report |

---

Copyright 2026 Olaf Yunus Laitinen Imanov and Contributors. Released under EUPL 1.2.
