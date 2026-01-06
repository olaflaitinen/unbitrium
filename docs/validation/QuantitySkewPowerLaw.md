# QuantitySkewPowerLaw Validation Report

## Overview

Quantity Skew with Power-Law Distribution partitions data such that client dataset sizes follow a power-law (Zipfian) distribution. This simulates realistic scenarios where a few clients have large datasets while many have small ones.

### Mathematical Formulation

Client $k$ receives $n_k$ samples where:

$$
n_k \propto k^{-\gamma}
$$

Normalized to ensure all samples assigned:

$$
n_k = \lfloor N \cdot \frac{k^{-\gamma}}{\sum_{j=1}^K j^{-\gamma}} \rfloor
$$

where:
- $N$ is total dataset size
- $\gamma > 0$ is the power-law exponent
- $K$ is number of clients

### Implementation Reference

The implementation is located at `src/unbitrium/partitioning/quantity_skew.py`.

---

## Invariants

### Invariant 1: Total Sample Conservation

All samples assigned (modulo rounding):

$$
\sum_{k=1}^K n_k \leq N
$$

**Verification**: Residual samples assigned to largest clients.

### Invariant 2: Monotonic Decrease

Sample counts decrease with client index:

$$
k_1 < k_2 \implies n_{k_1} \geq n_{k_2}
$$

**Verification**: Sorted order maintained.

### Invariant 3: Reduction to Uniform

When $\gamma = 0$, approaches uniform:

$$
\gamma = 0 \implies n_k = N/K
$$

**Verification**: Zero exponent produces equal sizes.

### Invariant 4: Extreme Skew

As $\gamma \to \infty$, first client dominates:

$$
\lim_{\gamma \to \infty} n_1 / N = 1
$$

**Verification**: Large $\gamma$ concentrates samples.

---

## Test Distributions

### Distribution 1: Zipf's Law ($\gamma = 1$)

**Configuration**:
- Dataset: $N = 60000$
- Clients: $K = 100$
- $\gamma = 1.0$

**Expected Behavior**:

| Client Rank | Expected Samples |
|-------------|------------------|
| 1 | ~12000 |
| 10 | ~1200 |
| 50 | ~240 |
| 100 | ~120 |

### Distribution 2: Moderate Skew

**Configuration**:
- $\gamma = 0.5$
- $K = 50$

**Expected Behavior**:
- Ratio of largest to smallest: ~7x
- Gini coefficient: ~0.3

### Distribution 3: Extreme Skew

**Configuration**:
- $\gamma = 2.0$
- $K = 100$

**Expected Behavior**:
- Client 1 has >50% of data
- Bottom 50% clients have <5% of data

### Distribution 4: Near-Uniform

**Configuration**:
- $\gamma = 0.1$
- $K = 100$

**Expected Behavior**:
- Size variation <2x
- Approximately IID in quantity

---

## Expected Behavior

### Gamma Interpretation

| $\gamma$ | Distribution Shape | Real-World Analogy |
|----------|-------------------|-------------------|
| 0 | Uniform | Controlled experiment |
| 0.5 | Mild skew | Enterprise devices |
| 1.0 | Zipf's law | Natural language, web |
| 1.5 | Strong skew | Social networks |
| 2.0+ | Extreme | Winner-take-all markets |

### Metric Ranges

| Metric | Range | Notes |
|--------|-------|-------|
| `gini_coefficient` | $[0, 1]$ | Inequality measure |
| `max_to_min_ratio` | $[1, \infty)$ | Size ratio |
| `median_samples` | $(0, N/K)$ | Median client size |
| `effective_clients` | $(0, K]$ | Clients with >1% of data |

---

## Edge Cases

### Edge Case 1: Single Client

**Input**: $K = 1$

**Expected Behavior**:
- Client receives all $N$ samples
- Power-law irrelevant

### Edge Case 2: More Clients Than Samples

**Input**: $K > N$

**Expected Behavior**:
- Some clients have zero samples
- Warning logged

### Edge Case 3: Zero Exponent

**Input**: $\gamma = 0$

**Expected Behavior**:
- Uniform distribution: $n_k = N/K$

### Edge Case 4: Negative Exponent

**Input**: $\gamma < 0$

**Expected Behavior**:
- Inverse power-law (larger indices get more)
- Should be handled or rejected

---

## Reproducibility

### Seed Configuration

```python
from unbitrium.partitioning import QuantitySkewPowerLaw

partitioner = QuantitySkewPowerLaw(
    gamma=1.0,
    num_clients=100,
    seed=42,
)
```

### Shuffling

**Important**: Power-law assigns by rank. For random assignment:

```python
partitioner = QuantitySkewPowerLaw(
    gamma=1.0,
    num_clients=100,
    shuffle=True,  # Randomize client-size mapping
    seed=42,
)
```

---

## Security Considerations

### Size Inference

Quantity skew reveals client data sizes:
- Large clients easily identified
- May correlate with organizational size

### Mitigations

1. Shuffle client indices before reporting
2. Use differential privacy on size reporting

---

## Complexity Analysis

### Time Complexity

$$
T = O(K + N)
$$

**Breakdown**:
- Size computation: $O(K)$
- Sample assignment: $O(N)$

### Space Complexity

$$
S = O(K + N)
$$

---

## References

1. Clauset, A., Shalizi, C. R., & Newman, M. E. (2009). Power-law distributions in empirical data. *SIAM Review*, 51(4), 661-703.

2. Li, Q., et al. (2022). Federated learning on non-IID data silos: An experimental study. In *ICDE*.

3. Luo, M., et al. (2021). Cost-effective federated learning design. In *INFOCOM*.

---

## Changelog

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2026-01-04 | Initial validation report |

---

Copyright 2026 Olaf Yunus Laitinen Imanov and Contributors. Released under EUPL 1.2.
