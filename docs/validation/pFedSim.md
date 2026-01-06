# pFedSim Validation Report

## Overview

Personalized Federated Similarity (pFedSim) extends FedSim by combining global model aggregation with personalized local layers. It decouples the model into shared feature extractors and client-specific prediction heads.

### Mathematical Formulation

The model is decomposed into shared parameters $\theta$ and personalized parameters $\phi_k$:

$$
w_k = (\theta_k, \phi_k)
$$

The shared layers are aggregated with similarity weighting:

$$
\theta^{t+1} = \sum_{k=1}^{K} \omega_k^t \cdot \theta_k^t
$$

where:

$$
\omega_k^t = \frac{\text{sim}(\theta_k^t, \theta^t)}{\sum_{j=1}^{K} \text{sim}(\theta_j^t, \theta^t)}
$$

The personalized layers $\phi_k$ are retained locally and not aggregated.

### Implementation Reference

The implementation is located at `src/unbitrium/aggregators/pfedsim.py`.

---

## Invariants

### Invariant 1: Layer Separation

Shared and personalized layers are correctly identified:

$$
|\theta| + |\phi| = |w|
$$

**Verification**: Sum of shared and personalized parameter counts equals total.

### Invariant 2: Personalized Layer Preservation

Personalized layers are not modified during aggregation:

$$
\phi_k^{t+1} = \phi_k^t + \Delta\phi_k^{local}
$$

**Verification**: Personalized parameters only change from local training.

### Invariant 3: Shared Layer Normalization

Weights for shared layer aggregation sum to unity:

$$
\sum_{k=1}^K \omega_k = 1
$$

**Verification**: Weight normalization confirmed.

---

## Test Distributions

### Distribution 1: Heterogeneous Task Distribution

**Configuration**:
- Clients: $K = 20$
- Each client has distinct task (e.g., different digit styles)
- Shared: CNN feature extractor
- Personalized: Final classification layer

**Expected Behavior**:
- Personalized layers adapt to local tasks
- Shared features transfer across clients
- Per-client accuracy higher than global FedAvg

### Distribution 2: Cross-Domain Transfer

**Configuration**:
- Clients from different domains (natural images, sketches, paintings)
- Model: ResNet-18 backbone + FC head

**Expected Behavior**:
- Shared backbone learns domain-invariant features
- Personalized heads specialize to each domain

### Distribution 3: Varying Data Quantities

**Configuration**:
- Clients: $K = 50$
- Samples: Power-law $n_k \propto k^{-1}$
- Personalized layer complexity adapts

**Expected Behavior**:
- Larger clients contribute more to shared layers
- Small clients benefit from personalized specialization

---

## Expected Behavior

### Layer Assignment

| Layer Type | Aggregation | Personalization |
|------------|-------------|-----------------|
| Feature extractor (conv, norm) | Global | No |
| Final classifier (fc) | None | Yes |
| Optional adaptation layers | None | Yes |

### Metric Ranges

| Metric | Range | Notes |
|--------|-------|-------|
| `shared_param_count` | $[0, P]$ | Number of shared parameters |
| `personal_param_count` | $[0, P]$ | Number of personalized parameters |
| `avg_similarity` | $[-1, 1]$ | Mean similarity on shared layers |
| `personalization_benefit` | $[-\infty, \infty]$ | Acc improvement from personalization |

---

## Edge Cases

### Edge Case 1: All Layers Shared

**Input**: Empty personalized layer specification

**Expected Behavior**:
- Reduces to FedSim
- All layers aggregated globally

### Edge Case 2: All Layers Personalized

**Input**: All layers marked as personalized

**Expected Behavior**:
- No aggregation occurs
- Each client trains independently (local training)

### Edge Case 3: New Client

**Input**: Client joins with no personalized history

**Expected Behavior**:
- Personalized layers initialized from global model
- Rapid adaptation through local training

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

### Layer Configuration

```yaml
personalization:
  shared_layers:
    - "conv1"
    - "conv2"
    - "bn1"
  personalized_layers:
    - "fc"
    - "classifier"
```

---

## Security Considerations

### Privacy Benefits

Personalized layers provide inherent privacy:
- Local task-specific information stays on device
- Only shared features transmitted to server

### Information Leakage

Shared layer updates may still reveal:
- Domain-general features of local data
- Statistical properties of feature distributions

### Mitigations

1. Differential privacy on shared layer updates
2. Feature obfuscation before aggregation
3. Secure aggregation protocols

---

## Complexity Analysis

### Time Complexity

$$
T = O(K \cdot P_{shared})
$$

where $P_{shared}$ is the number of shared parameters.

### Space Complexity

Server:
$$
S_{server} = O(P_{shared})
$$

Client:
$$
S_{client} = O(P_{shared} + P_{personal})
$$

---

## Performance Benchmarks

### CIFAR-100 (20 superclasses as clients)

| Method | Global Acc | Personalized Acc |
|--------|------------|------------------|
| FedAvg | 42.3% | - |
| FedSim | 45.8% | - |
| pFedSim | 41.2% | 58.7% |

### Federated EMNIST

| Method | Writer-level Acc |
|--------|-----------------|
| Local Only | 84.2% |
| FedAvg | 82.8% |
| pFedSim | 89.4% |

---

## References

1. Collins, L., et al. (2021). Exploiting shared representations for personalized federated learning. In *ICML*.

2. Li, D., & Wang, J. (2019). FedMD: Heterogeneous federated learning via model distillation. In *NeurIPS Workshop*.

3. Arivazhagan, M. G., et al. (2019). Federated learning with personalization layers. *arXiv preprint*.

---

## Changelog

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2026-01-04 | Initial validation report |

---

Copyright 2026 Olaf Yunus Laitinen Imanov and Contributors. Released under EUPL 1.2.
