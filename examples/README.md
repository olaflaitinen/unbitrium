# Unbitrium Examples

This directory contains reproducible example scripts demonstrating
core Unbitrium functionality.

## Scripts

### quickstart.py

Minimal end-to-end federated learning simulation using FedAvg aggregation
on synthetically partitioned data.

```bash
python examples/quickstart.py
```

### dirichlet_partitioning.py

Demonstrates various non-IID partitioning strategies including Dirichlet,
Mixture-of-Dirichlet-Multinomials (MoDM), quantity skew, and entropy-controlled
partitioning. Generates visualization PNG files.

```bash
python examples/dirichlet_partitioning.py
```

### similarity_aggregation.py

Compares FedAvg, FedSim, and pFedSim aggregation strategies under
heterogeneous data distributions.

```bash
python examples/similarity_aggregation.py
```

### benchmark_run.py

Full benchmark harness demonstration with experiment provenance tracking
and standardized result artifacts.

```bash
python examples/benchmark_run.py
```

## Requirements

All examples require the Unbitrium package to be installed:

```bash
pip install unbitrium
```

Or install from source:

```bash
pip install -e .
```
