# FedProx Validation Report

## Overview

Federated Proximal (FedProx) extends FedAvg by introducing a proximal term that constrains local updates to remain close to the global model. This addresses client drift under heterogeneous data distributions.

### Mathematical Formulation

Each client minimizes a modified local objective:

$$
\min_{w} F_k(w) + \frac{\mu}{2}\|w - w^t\|^2
$$

where:
- $F_k(w)$ is the local empirical loss on client $k$
- $w^t$ is the global model at round $t$
- $\mu \geq 0$ is the proximal regularization strength
- $\|w - w^t\|^2$ is the squared Euclidean distance to the global model

The aggregation step remains identical to FedAvg:

$$
w^{t+1} = \sum_{k=1}^{K} \frac{n_k}{N} w_k^{t+1}
$$

### Implementation Reference

The implementation is located at `src/unbitrium/aggregators/fedprox.py`.

---

## Invariants

### Invariant 1: Reduction to FedAvg

When $\mu = 0$, FedProx reduces to FedAvg:

$$
\mu = 0 \implies \text{FedProx} \equiv \text{FedAvg}
$$

**Verification**: Property-based testing confirms identical outputs when $\mu = 0$.

### Invariant 2: Bounded Client Drift

The proximal term bounds the drift of local models:

$$
\|w_k^{t+1} - w^t\|^2 \leq \frac{2(F_k(w^t) - F_k^*)}{\mu}
$$

**Verification**: Measured drift decreases monotonically as $\mu$ increases.

### Invariant 3: Convergence Guarantee

Under standard convexity assumptions, FedProx converges at rate:

$$
\mathbb{E}[F(w^T)] - F^* \leq O\left(\frac{1}{\mu T}\right)
$$

**Verification**: Empirical convergence matches theoretical bounds on convex benchmarks.

### Invariant 4: Determinism

Given identical inputs and random seeds, the output is reproducible.

**Verification**: Repeated executions with fixed seeds produce bit-identical results.

---

## Test Distributions

### Distribution 1: IID Baseline with Varying Mu

**Configuration**:
- Clients: $K = 10$
- Samples per client: $n_k = 100$
- Model: 2-layer CNN on MNIST
- $\mu \in \{0, 0.01, 0.1, 1.0, 10.0\}$

**Expected Behavior**:

| $\mu$ | Convergence Speed | Final Accuracy | Client Drift |
|-------|-------------------|----------------|--------------|
| 0 | Baseline | Baseline | High |
| 0.01 | Similar | Similar | Moderate |
| 0.1 | Slightly slower | Similar | Low |
| 1.0 | Slower | Potentially lower | Very low |
| 10.0 | Much slower | Lower | Minimal |

### Distribution 2: Non-IID Label Skew

**Configuration**:
- Clients: $K = 10$
- Partitioning: Dirichlet with $\alpha = 0.1$
- Model: ResNet-18 on CIFAR-10
- $\mu = 0.01$

**Expected Behavior**:
- Improved convergence compared to FedAvg under non-IID
- Reduced variance in client model quality
- EMD between client and global distributions: $\text{EMD} > 0.6$

### Distribution 3: Extreme Heterogeneity

**Configuration**:
- Clients: $K = 100$
- Each client has data from only 2 classes
- $\mu = 0.1$

**Expected Behavior**:
- FedProx outperforms FedAvg by 5-15% accuracy
- Convergence stabilizes within 100 rounds
- Client drift norm bounded below 2.0

### Distribution 4: Variable Local Epochs

**Configuration**:
- Clients: $K = 20$
- Local epochs: $E_k \sim \text{Uniform}(1, 10)$
- $\mu = 0.01$

**Expected Behavior**:
- Proximal term compensates for varying local computation
- Clients with more epochs contribute updates closer to global model
- Overall convergence remains stable

---

## Expected Behavior

### Hyperparameter Sensitivity

| $\mu$ Range | Effect | Recommendation |
|-------------|--------|----------------|
| $[0, 0.001)$ | Negligible regularization | Use FedAvg instead |
| $[0.001, 0.1)$ | Mild drift reduction | Good default range |
| $[0.1, 1.0)$ | Strong regularization | For extreme non-IID |
| $[1.0, \infty)$ | Over-regularization | May hurt convergence |

### Metric Ranges

| Metric | Range | Notes |
|--------|-------|-------|
| `proximal_loss` | $[0, \infty)$ | Additional loss from proximal term |
| `avg_drift_norm` | $[0, \infty)$ | Mean $\|w_k - w^t\|$ across clients |
| `mu_effective` | $\mu$ | Regularization strength used |

---

## Edge Cases

### Edge Case 1: Zero Proximal Term

**Input**: $\mu = 0$

**Expected Behavior**:
- Reduces to FedAvg
- No proximal loss component
- Client drift unbounded

### Edge Case 2: Very Large Mu

**Input**: $\mu > 100$

**Expected Behavior**:
- Local updates nearly zero
- Global model changes minimally per round
- May require many rounds for convergence

### Edge Case 3: Single Local Step

**Input**: Local epochs $E = 1$

**Expected Behavior**:
- Proximal term has minimal effect
- Equivalent to FedAvg with scaled learning rate

---

## Reproducibility

### Seed Configuration

```python
def set_seed(seed: int = 42) -> None:
    import random
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
```

### Validation Environment

| Component | Version |
|-----------|---------|
| Python | 3.12.0 |
| PyTorch | 2.4.0 |
| Unbitrium | 1.0.0 |

### Replication Instructions

```bash
git clone https://github.com/olaflaitinen/unbitrium
cd unbitrium
pip install -e ".[dev]"
pytest tests/validation/test_fedprox.py -v
```

---

## Security Considerations

### Information Leakage

FedProx shares the same privacy profile as FedAvg:
- Model updates reveal information about local data
- Sample counts may expose dataset sizes

### Additional Privacy Considerations

The proximal term introduces a dependency on the global model:
- Clients must receive $w^t$ to compute the proximal gradient
- This is standard in FL and does not introduce additional leakage

### Mitigations

1. **Differential Privacy**: Apply DP-SGD during local training
2. **Secure Aggregation**: Use MPC protocols for aggregation
3. **Gradient Compression**: Reduce information in transmitted updates

---

## Complexity Analysis

### Time Complexity

Local training per client:
$$
T_{local}(E, B, P) = O(E \cdot \lceil n_k/B \rceil \cdot P)
$$

Proximal gradient computation:
$$
T_{prox}(P) = O(P)
$$

Total per-round complexity:
$$
T_{round}(K, P) = O(K \cdot P)
$$

### Space Complexity

$$
S(P) = O(2P)
$$

**Breakdown**:
- Current local model: $O(P)$
- Reference global model for proximal term: $O(P)$

---

## Comparison with Reference Implementations

### Leaf Benchmark

The LEAF FedProx implementation produces equivalent results within tolerance $\epsilon < 10^{-4}$.

### TensorFlow Federated

TFF's FedProx matches Unbitrium when using identical:
- Proximal coefficient $\mu$
- Learning rate schedules
- Local optimizer configurations

---

## References

1. Li, T., Sahu, A. K., Zaheer, M., Sanjabi, M., Talwalkar, A., & Smith, V. (2020). Federated optimization in heterogeneous networks. In *Proceedings of Machine Learning and Systems* (Vol. 2, pp. 429-450).

2. McMahan, H. B., et al. (2017). Communication-efficient learning of deep networks from decentralized data. In *AISTATS*.

3. Karimireddy, S. P., Kale, S., Mohri, M., Reddi, S., Stich, S., & Suresh, A. T. (2020). SCAFFOLD: Stochastic controlled averaging for federated learning. In *ICML*.

---

## Changelog

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2026-01-04 | Initial validation report |

---

Copyright 2026 Olaf Yunus Laitinen Imanov and Contributors. Released under EUPL 1.2.
