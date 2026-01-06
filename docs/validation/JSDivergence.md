# JSDivergence Validation Report

## Overview

Jensen-Shannon Divergence (JSD) is a symmetric, bounded measure of similarity between probability distributions. It addresses the asymmetry of KL divergence and is always finite.

### Mathematical Formulation

$$
\text{JS}(p \| q) = \frac{1}{2} \text{KL}(p \| m) + \frac{1}{2} \text{KL}(q \| m)
$$

where $m = \frac{1}{2}(p + q)$ is the mixture distribution.

Expanding the KL terms:

$$
\text{JS}(p \| q) = \frac{1}{2} \sum_i p_i \log\frac{2p_i}{p_i + q_i} + \frac{1}{2} \sum_i q_i \log\frac{2q_i}{p_i + q_i}
$$

The square root $\sqrt{\text{JS}(p \| q)}$ is a proper metric (satisfies triangle inequality).

### Implementation Reference

The implementation is located at `src/unbitrium/metrics/distribution.py`.

---

## Invariants

### Invariant 1: Non-negativity

$$
\text{JS}(p \| q) \geq 0
$$

**Verification**: All computed values non-negative.

### Invariant 2: Symmetry

$$
\text{JS}(p \| q) = \text{JS}(q \| p)
$$

**Verification**: Order-independent.

### Invariant 3: Identity

$$
\text{JS}(p \| p) = 0
$$

**Verification**: Same distribution yields zero.

### Invariant 4: Bounded Range

Using natural logarithm:

$$
0 \leq \text{JS}(p \| q) \leq \ln 2
$$

Using base-2 logarithm:

$$
0 \leq \text{JS}(p \| q) \leq 1
$$

**Verification**: All values within bounds.

---

## Test Distributions

### Distribution 1: Identical Distributions

**Input**: $p = q$

**Expected Output**: JS = 0

### Distribution 2: Non-overlapping Distributions

**Input**:
- $p$: All mass on class 0
- $q$: All mass on class 1

**Expected Output**: JS = $\ln 2 \approx 0.693$ (natural log)

### Distribution 3: Partial Overlap

**Input**:
- $p = (0.5, 0.5, 0, 0)$
- $q = (0, 0, 0.5, 0.5)$

**Expected Output**: JS = $\ln 2$

### Distribution 4: Near-Identical

**Input**: Small perturbation from uniform

**Expected Behavior**: JS close to zero

---

## Expected Behavior

### JS Interpretation Guide

| JS (base e) | JS (base 2) | Heterogeneity |
|-------------|-------------|---------------|
| 0.0 - 0.1 | 0.0 - 0.14 | Minimal |
| 0.1 - 0.3 | 0.14 - 0.43 | Low |
| 0.3 - 0.5 | 0.43 - 0.72 | Moderate |
| 0.5 - 0.693 | 0.72 - 1.0 | High |

### Comparison with KL Divergence

| Property | KL | JS |
|----------|-----|-----|
| Symmetric | No | Yes |
| Bounded | No | Yes |
| Metric | No | sqrt(JS) is |
| Handles zeros | Problematic | Safe |

---

## Edge Cases

### Edge Case 1: Zero Probability Classes

**Input**: $p_i = 0$ for some $i$

**Expected Behavior**:
- Computed correctly using convention $0 \log 0 = 0$
- No numerical issues

### Edge Case 2: Sparse Distributions

**Input**: Most probability mass on few classes

**Expected Behavior**:
- Stable computation
- Correct handling of near-zero entries

### Edge Case 3: Single Class

**Input**: $C = 1$

**Expected Behavior**:
- $p = q = 1$
- JS = 0

---

## Reproducibility

### Usage Example

```python
from unbitrium.metrics import JSDivergence

metric = JSDivergence(base='e')  # or 'base2'

p = np.array([0.3, 0.3, 0.4])
q = np.array([0.5, 0.3, 0.2])

js = metric.compute(p, q)
print(f"JS Divergence: {js:.4f}")
```

### Batch Computation

```python
# Compute for all clients vs global
global_dist = compute_label_distribution(global_data)
js_values = []

for client_data in partitions:
    client_dist = compute_label_distribution(client_data)
    js = metric.compute(client_dist, global_dist)
    js_values.append(js)
```

---

## Security Considerations

### Information Content

JS divergence reveals:
- Distribution similarity
- Clustering structure

### Mitigations

Same as EMD: aggregate reporting, differential privacy.

---

## Complexity Analysis

### Time Complexity

$$
T = O(C)
$$

Linear in number of classes.

### Space Complexity

$$
S = O(C)
$$

---

## Relationship to Other Metrics

### Information Theoretic View

JS is the mutual information between:
- A binary indicator variable (which distribution)
- The sampled outcome

### KL Divergence Bound

$$
\text{JS}(p \| q) \leq \frac{1}{2}\text{KL}(p \| q) + \frac{1}{2}\text{KL}(q \| p)
$$

### Total Variation Bound

$$
\text{JS}(p \| q) \leq \ln 2 \cdot \text{TV}(p, q)
$$

where TV is total variation distance.

---

## References

1. Lin, J. (1991). Divergence measures based on the Shannon entropy. *IEEE Transactions on Information Theory*, 37(1), 145-151.

2. Endres, D. M., & Schindelin, J. E. (2003). A new metric for probability distributions. *IEEE Transactions on Information Theory*, 49(7), 1858-1860.

3. Fuglede, B., & Topsoe, F. (2004). Jensen-Shannon divergence and Hilbert space embedding. In *IEEE International Symposium on Information Theory*.

---

## Changelog

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2026-01-04 | Initial validation report |

---

Copyright 2026 Olaf Yunus Laitinen Imanov and Contributors. Released under EUPL 1.2.
