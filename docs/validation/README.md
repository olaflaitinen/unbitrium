# Validation Documentation

This directory contains validation reports for all core components of Unbitrium. Each report documents the formal verification methodology, synthetic test distributions, expected behavior ranges, and reproducibility protocols.

---

## Purpose

Validation documentation serves three critical functions in the Unbitrium framework:

1. **Correctness Verification**: Establishes that implementations match their formal mathematical specifications.
2. **Reproducibility Assurance**: Provides explicit seeds, configurations, and expected outputs for independent replication.
3. **Performance Baselining**: Documents expected metric ranges under controlled conditions.

---

## Methodology

### Formal Verification Protocol

Each component undergoes a structured validation process:

1. **Specification Review**: Mathematical formulation is reviewed against source publications.
2. **Property Identification**: Key invariants and edge cases are enumerated.
3. **Test Distribution Design**: Synthetic data distributions with known properties are constructed.
4. **Implementation Testing**: Property-based and deterministic tests verify correctness.
5. **Benchmark Comparison**: Where applicable, outputs are compared against reference implementations.

### Reproducibility Standards

All validation experiments adhere to:

- **Deterministic Seeding**: All random number generators are seeded with documented values.
- **Version Pinning**: Library versions are recorded in manifests.
- **Hardware Neutrality**: Tests are designed to produce consistent results across platforms.

---

## Document Structure

Each validation report follows a standardized structure:

| Section | Description |
|---------|-------------|
| **Overview** | Component purpose and mathematical formulation |
| **Invariants** | Properties that must hold for all valid inputs |
| **Test Distributions** | Synthetic data configurations with known characteristics |
| **Expected Behavior** | Metric ranges and output characteristics |
| **Edge Cases** | Boundary conditions and failure modes |
| **Reproducibility** | Seeds, configurations, and replication instructions |
| **Security Considerations** | Information leakage analysis and mitigations |
| **References** | Source publications and related work |

---

## Component Index

### Aggregators

| Component | File | Status |
|-----------|------|--------|
| FedAvg | [FedAvg.md](FedAvg.md) | Validated |
| FedProx | [FedProx.md](FedProx.md) | Validated |
| FedDyn | [FedDyn.md](FedDyn.md) | Validated |
| FedSim | [FedSim.md](FedSim.md) | Validated |
| pFedSim | [pFedSim.md](pFedSim.md) | Validated |
| FedCM | [FedCM.md](FedCM.md) | Validated |
| AFL-DCS | [AFL-DCS.md](AFL-DCS.md) | Validated |
| FedAdam | [FedAdam.md](FedAdam.md) | Validated |
| TrimmedMean | [TrimmedMean.md](TrimmedMean.md) | Validated |
| Krum | [Krum.md](Krum.md) | Validated |

### Partitioners

| Component | File | Status |
|-----------|------|--------|
| DirichletLabelSkew | [DirichletLabelSkew.md](DirichletLabelSkew.md) | Validated |
| MoDM | [MoDM.md](MoDM.md) | Validated |
| QuantitySkewPowerLaw | [QuantitySkewPowerLaw.md](QuantitySkewPowerLaw.md) | Validated |
| FeatureShiftClustering | [FeatureShiftClustering.md](FeatureShiftClustering.md) | Validated |
| EntropyControlledPartition | [EntropyControlledPartition.md](EntropyControlledPartition.md) | Validated |

### Metrics

| Component | File | Status |
|-----------|------|--------|
| EMDLabelDistance | [EMDLabelDistance.md](EMDLabelDistance.md) | Validated |
| JSDivergence | [JSDivergence.md](JSDivergence.md) | Validated |
| GradientVariance | [GradientVariance.md](GradientVariance.md) | Validated |
| NMIRepresentations | [NMIRepresentations.md](NMIRepresentations.md) | Validated |
| ClientDriftNorm | [ClientDriftNorm.md](ClientDriftNorm.md) | Validated |

---

## Running Validation Tests

To execute the full validation suite:

```bash
# Run all validation tests
pytest tests/validation/ -v --tb=short

# Run with coverage reporting
pytest tests/validation/ --cov=src/unbitrium --cov-report=html

# Run specific component validation
pytest tests/validation/test_aggregators.py -k "FedAvg"
```

---

## Citation

If you use Unbitrium's validation methodology in your research, please cite:

```bibtex
@software{unbitrium_validation2026,
  author       = {Laitinen Imanov, Olaf Yunus and Contributors},
  title        = {Unbitrium Validation Framework},
  year         = {2026},
  publisher    = {GitHub},
  url          = {https://github.com/olaflaitinen/unbitrium/docs/validation}
}
```

---

## Contributing

Contributions to validation documentation are welcome. Please see [CONTRIBUTING.md](../../CONTRIBUTING.md) for guidelines on:

- Adding new component validations
- Extending test distributions
- Reporting validation failures

---

Copyright 2026 Olaf Yunus Laitinen Imanov and Contributors. Released under EUPL 1.2.
