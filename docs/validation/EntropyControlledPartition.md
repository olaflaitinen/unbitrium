# EntropyControlledPartition Validation Report

## Overview

Entropy-Controlled Partition creates non-IID distributions with precise control over partition "hardness" measured by label entropy. This enables systematic benchmarking across controlled heterogeneity levels.

### Mathematical Formulation

Label entropy for client $k$:

$$
H(p_k) = -\sum_{c=1}^C p_{k,c} \log p_{k,c}
$$

where $p_{k,c}$ is the proportion of class $c$ in client $k$'s data.

The partitioner targets a specific entropy level:

$$
H_{target} \in [0, \log C]
$$

where:
- $H = 0$: Single-class client (maximum non-IID)
- $H = \log C$: Uniform distribution (IID)

### Implementation Reference

The implementation is located at `src/unbitrium/partitioning/entropy_controlled.py`.

---

## Invariants

### Invariant 1: Entropy Bounds

Achieved entropy within specified tolerance:

$$
|H(p_k) - H_{target}| \leq \epsilon
$$

**Verification**: Entropy constraint satisfied for all clients.

### Invariant 2: Valid Probability

Label proportions form valid distributions:

$$
\sum_{c=1}^C p_{k,c} = 1, \quad p_{k,c} \geq 0
$$

**Verification**: Proportions normalized.

### Invariant 3: Monotonic Hardness

Lower target entropy implies higher heterogeneity:

$$
H_1 < H_2 \implies \text{EMD}(p_1, p_{uniform}) > \text{EMD}(p_2, p_{uniform})
$$

**Verification**: EMD correlates inversely with entropy.

### Invariant 4: Reproducibility

Fixed seed produces identical partitions.

---

## Test Distributions

### Distribution 1: Low Entropy (Hard)

**Configuration**:
- Target entropy: $H = 0.5$ (vs $\log 10 = 2.3$)
- Dataset: CIFAR-10
- Clients: $K = 50$

**Expected Behavior**:
- Clients have 1-2 dominant classes
- High EMD values (0.7-0.9)
- Slow convergence expected

### Distribution 2: Medium Entropy

**Configuration**:
- Target entropy: $H = 1.5$
- Tolerance: $\epsilon = 0.1$

**Expected Behavior**:
- Clients have 3-5 significant classes
- Moderate EMD (0.3-0.5)
- Balanced heterogeneity

### Distribution 3: High Entropy (Easy)

**Configuration**:
- Target entropy: $H = 2.2$ (near maximum)

**Expected Behavior**:
- Near-uniform label distributions
- Low EMD (<0.1)
- IID-like behavior

### Distribution 4: Sweep Experiment

**Configuration**:
- $H_{target} \in \{0.3, 0.5, 1.0, 1.5, 2.0, 2.3\}$
- Same dataset and clients

**Expected Behavior**:
- Systematic variation in heterogeneity
- Reproducible benchmark conditions

---

## Expected Behavior

### Entropy-Heterogeneity Mapping

| Entropy $H$ | $H / H_{max}$ | Hardness | FL Impact |
|-------------|---------------|----------|-----------|
| 0.0 - 0.5 | 0 - 20% | Very hard | Severe degradation |
| 0.5 - 1.0 | 20 - 40% | Hard | Significant degradation |
| 1.0 - 1.5 | 40 - 65% | Medium | Moderate impact |
| 1.5 - 2.0 | 65 - 85% | Easy | Minor impact |
| 2.0 - 2.3 | 85 - 100% | Very easy | Near-IID |

### Metric Ranges

| Metric | Range | Notes |
|--------|-------|-------|
| `target_entropy` | $[0, \log C]$ | Requested entropy |
| `achieved_entropy` | $[0, \log C]$ | Actual mean entropy |
| `entropy_variance` | $[0, \infty)$ | Variance across clients |
| `entropy_tolerance` | $(0, 1]$ | Acceptable deviation |

---

## Edge Cases

### Edge Case 1: Minimum Entropy ($H = 0$)

**Input**: Target $H = 0$

**Expected Behavior**:
- Each client has single class
- Perfect non-IID
- May not be achievable if $K > C$

### Edge Case 2: Maximum Entropy

**Input**: Target $H = \log C$

**Expected Behavior**:
- Uniform distributions
- IID partitioning
- Trivially achievable

### Edge Case 3: Infeasible Target

**Input**: Constraints cannot be satisfied

**Expected Behavior**:
- Best-effort partitioning
- Warning logged with achieved entropy
- Fallback to closest feasible

### Edge Case 4: Small Dataset

**Input**: Few samples per client

**Expected Behavior**:
- Discrete entropy effects
- May not achieve target precisely

---

## Reproducibility

### Seed Configuration

```python
from unbitrium.partitioning import EntropyControlledPartition

partitioner = EntropyControlledPartition(
    target_entropy=1.0,
    tolerance=0.1,
    num_clients=100,
    seed=42,
)
```

### Verification

```python
# Verify achieved entropy
from scipy.stats import entropy

for client_data in partitions:
    labels = [y for x, y in client_data]
    label_counts = np.bincount(labels, minlength=num_classes)
    label_probs = label_counts / label_counts.sum()
    client_entropy = entropy(label_probs, base=np.e)
    assert abs(client_entropy - target_entropy) < tolerance
```

---

## Security Considerations

### Control Implications

Entropy control enables:
- Precise adversarial testing
- Worst-case scenario simulation

### Mitigations

1. Use for benchmarking only
2. Document entropy levels in experiment reports

---

## Complexity Analysis

### Time Complexity

$$
T = O(N \cdot I)
$$

where $I$ is iterations to achieve target entropy.

### Space Complexity

$$
S = O(K \cdot C + N)
$$

---

## Algorithm Details

### Entropy Targeting Process

1. Initialize with Dirichlet samples
2. Compute current entropy per client
3. Adjust proportions toward target:
   - If $H_{current} > H_{target}$: Concentrate on fewer classes
   - If $H_{current} < H_{target}$: Spread across more classes
4. Re-assign samples to match adjusted proportions
5. Iterate until convergence

### Optimization Formulation

$$
\min_{p_k} \sum_k (H(p_k) - H_{target})^2
$$

subject to probability simplex constraints.

---

## References

1. FedSym Paper (2021). Entropy-based heterogeneity benchmarking in federated learning.

2. Wang, J., et al. (2020). Tackling the objective inconsistency problem in heterogeneous federated optimization. In *NeurIPS*.

3. Li, Q., et al. (2022). Federated learning on non-IID data silos: An experimental study. In *ICDE*.

---

## Changelog

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2026-01-04 | Initial validation report |

---

Copyright 2026 Olaf Yunus Laitinen Imanov and Contributors. Released under EUPL 1.2.
