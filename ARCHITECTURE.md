# Architecture

This document describes the high-level architecture of Unbitrium, a production-grade federated learning simulation and benchmarking platform.

---

## Table of Contents

1. [Overview](#overview)
2. [Design Principles](#design-principles)
3. [System Architecture](#system-architecture)
4. [Module Overview](#module-overview)
5. [Data Flow](#data-flow)
6. [Class Hierarchies](#class-hierarchies)
7. [Extension Points](#extension-points)
8. [Dependencies](#dependencies)
9. [Performance Considerations](#performance-considerations)

---

## Overview

Unbitrium is a modular federated learning simulation and benchmarking platform designed for reproducible research. The architecture prioritizes:

- **Modularity**: Independent, composable components
- **Extensibility**: Easy addition of new algorithms
- **Reproducibility**: Deterministic execution and provenance
- **Research Focus**: Heterogeneity measurement and analysis
- **Type Safety**: Full type annotations with strict mypy
- **Performance**: Vectorized operations and GPU acceleration

---

## Design Principles

### Separation of Concerns

Each module handles a specific aspect of federated learning:

| Module | Responsibility |
|--------|----------------|
| `partitioning` | Data distribution synthesis |
| `aggregators` | Model update combination |
| `metrics` | Heterogeneity quantification |
| `simulation` | Client/server orchestration |
| `privacy` | DP mechanisms |
| `systems` | Device/network modeling |
| `datasets` | Dataset loading and registry |
| `bench` | Benchmarking infrastructure |

### Composition Over Inheritance

Components are designed to be composed:

```python
# Compose partitioner, aggregator, and metrics
partitioner = DirichletPartitioner(num_clients=100, alpha=0.5)
aggregator = FedAvg()
metrics = [compute_emd, compute_label_entropy]

# Use in simulation
engine = SimulationEngine(config, aggregator, model, datasets)
```

### Stateless Functions

Metric computations are pure functions:

```python
# No side effects, deterministic output
emd = compute_emd(labels, client_indices)
entropy = compute_label_entropy(labels, client_indices)
```

---

## System Architecture

### High-Level Architecture

```mermaid
graph TB
    subgraph "Simulation Engine"
        ENG[Engine]
        EVT[Event System]
        PRV[Provenance Tracker]
        RNG[RNG Manager]
        LOG[Logger]
    end

    subgraph "Data Layer"
        DS[Datasets]
        PART[Partitioners]
        LOAD[Loaders]
    end

    subgraph "Aggregation Layer"
        FAVG[FedAvg]
        FPROX[FedProx]
        FSIM[FedSim]
        KRUM[Krum]
    end

    subgraph "Metrics Layer"
        EMD[EMD]
        ENT[Entropy]
        VAR[Variance]
        NMI[NMI]
    end

    subgraph "Privacy Layer"
        DP[DP Mechanisms]
        SEC[Secure Aggregation]
        CLIP[Gradient Clipping]
    end

    subgraph "Systems Layer"
        DEV[Device Model]
        ENERGY[Energy Model]
        NET[Network Model]
    end

    subgraph "Benchmarking Layer"
        RUN[Runner]
        CFG[Config]
        RPT[Reports]
    end

    ENG --> EVT
    ENG --> PRV
    ENG --> RNG
    ENG --> LOG

    DS --> PART
    PART --> LOAD

    ENG --> DS
    ENG --> FAVG
    ENG --> FPROX
    ENG --> FSIM
    ENG --> KRUM

    ENG --> EMD
    ENG --> ENT
    ENG --> VAR
    ENG --> NMI

    FAVG --> DP
    FAVG --> SEC
    FAVG --> CLIP

    ENG --> DEV
    ENG --> ENERGY
    ENG --> NET

    ENG --> RUN
    RUN --> CFG
    RUN --> RPT
```

### Layer Dependencies

```mermaid
graph LR
    subgraph "Application Layer"
        BENCH[Benchmarking]
    end

    subgraph "Orchestration Layer"
        SIM[Simulation Engine]
    end

    subgraph "Algorithm Layer"
        AGG[Aggregators]
        MET[Metrics]
        PRIV[Privacy]
    end

    subgraph "Data Layer"
        DATA[Datasets]
        PART[Partitioning]
    end

    subgraph "Infrastructure Layer"
        CORE[Core]
        SYS[Systems]
    end

    BENCH --> SIM
    SIM --> AGG
    SIM --> MET
    SIM --> PRIV
    SIM --> DATA
    SIM --> PART
    AGG --> CORE
    MET --> CORE
    DATA --> CORE
    PART --> CORE
    PRIV --> CORE
    SIM --> SYS
```

---

## Module Overview

### Core (`unbitrium.core`)

Central infrastructure components.

| Component | Purpose |
|-----------|---------|
| `SimulationEngine` | Orchestrates FL rounds |
| `EventSystem` | Publish-subscribe events |
| `ProvenanceTracker` | Experiment metadata |
| `RNGManager` | Deterministic randomness |
| `Logger` | Structured logging |

### Partitioning (`unbitrium.partitioning`)

Data distribution strategies.

| Strategy | Description |
|----------|-------------|
| `DirichletPartitioner` | Label skew via Dirichlet |
| `MoDMPartitioner` | Mixture-of-Dirichlet |
| `QuantitySkewPartitioner` | Power-law sizes |
| `EntropyControlledPartitioner` | Target entropy |
| `FeatureShiftPartitioner` | Feature clustering |

### Aggregators (`unbitrium.aggregators`)

Model combination algorithms.

| Algorithm | Description |
|-----------|-------------|
| `FedAvg` | Weighted average |
| `FedProx` | Proximal regularization |
| `FedSim` | Similarity weighting |
| `PFedSim` | Personalized similarity |
| `FedDyn` | Dynamic regularization |
| `FedCM` | Client momentum |
| `FedAdam` | Server-side Adam |
| `Krum` | Byzantine-robust |
| `TrimmedMean` | Coordinate trimming |

### Metrics (`unbitrium.metrics`)

Heterogeneity quantification.

| Category | Metrics |
|----------|---------|
| Distribution | EMD, KL, JS, Total Variation |
| Label | Entropy, Imbalance Ratio |
| Gradient | Variance, Drift Norm |
| Representation | NMI, CKA |
| System | Latency, Throughput |

### Privacy (`unbitrium.privacy`)

Privacy-preserving mechanisms.

| Component | Purpose |
|-----------|---------|
| `GaussianMechanism` | (ε,δ)-DP noise |
| `LaplaceMechanism` | ε-DP noise |
| `SecureAggregation` | Simulation interface |
| `clip_gradients` | Sensitivity bounding |

### Systems (`unbitrium.systems`)

Device and network modeling.

| Component | Purpose |
|-----------|---------|
| `Device` | Compute/memory simulation |
| `EnergyModel` | Energy consumption |
| `Network` | Latency/bandwidth |

### Benchmark (`unbitrium.bench`)

Standardized experimentation.

| Component | Purpose |
|-----------|---------|
| `BenchmarkRunner` | Experiment execution |
| `BenchmarkConfig` | Configuration schema |
| `Artifacts` | Result storage |
| `Reports` | Markdown generation |

---

## Data Flow

### Simulation Flow

```mermaid
flowchart TD
    A[1. Load Configuration] --> B[2. Initialize Components]
    B --> C[Create Global Model]
    B --> D[Initialize Partitioner]
    B --> E[Setup Aggregator]

    C --> F[3. Training Loop]
    D --> F
    E --> F

    F --> G[Client Selection]
    G --> H[Model Broadcast]
    H --> I[Local Training]
    I --> J[Update Collection]
    J --> K[Aggregation]
    K --> L[Metric Computation]
    L --> M{More Rounds?}

    M -->|Yes| G
    M -->|No| N[4. Finalization]

    N --> O[Save Checkpoints]
    N --> P[Final Metrics]
    N --> Q[Generate Report]
```

### Aggregation Flow

```mermaid
sequenceDiagram
    participant Clients
    participant Aggregator
    participant GlobalModel

    Clients->>Aggregator: Send updates [update_1, update_2, ...]

    Note over Aggregator: 1. Validate updates
    Note over Aggregator: 2. Compute weights
    Note over Aggregator: 3. Aggregate parameters
    Note over Aggregator: 4. Compute metrics

    Aggregator->>GlobalModel: Updated model weights
    Aggregator->>Clients: Aggregation metrics
```

### Privacy-Preserving Flow

```mermaid
flowchart LR
    A[Raw Gradients] --> B[Gradient Clipping]
    B --> C[Noise Addition]
    C --> D[Aggregation]
    D --> E[Privacy Accounting]
    E --> F[Model Update]
```

---

## Class Hierarchies

### Aggregator Hierarchy

```mermaid
classDiagram
    class Aggregator {
        <<abstract>>
        +aggregate(updates, model) tuple
    }

    class FedAvg {
        +aggregate(updates, model) tuple
    }

    class FedProx {
        -mu: float
        +aggregate(updates, model) tuple
    }

    class FedSim {
        -temperature: float
        +aggregate(updates, model) tuple
    }

    class Krum {
        -num_byzantine: int
        +aggregate(updates, model) tuple
    }

    Aggregator <|-- FedAvg
    Aggregator <|-- FedProx
    Aggregator <|-- FedSim
    Aggregator <|-- Krum
```

### Partitioner Hierarchy

```mermaid
classDiagram
    class Partitioner {
        <<abstract>>
        +partition(labels) dict
    }

    class DirichletPartitioner {
        -num_clients: int
        -alpha: float
        +partition(labels) dict
    }

    class QuantitySkewPartitioner {
        -num_clients: int
        -power: float
        +partition(labels) dict
    }

    class FeatureShiftPartitioner {
        -num_clients: int
        -num_clusters: int
        +partition_features(features, labels) dict
    }

    Partitioner <|-- DirichletPartitioner
    Partitioner <|-- QuantitySkewPartitioner
    Partitioner <|-- FeatureShiftPartitioner
```

---

## Extension Points

### Custom Aggregator

Implement the `Aggregator` base class:

```python
from unbitrium.aggregators.base import Aggregator

class CustomAggregator(Aggregator):
    def aggregate(
        self,
        updates: list[dict[str, Any]],
        current_global_model: torch.nn.Module,
    ) -> tuple[torch.nn.Module, dict[str, float]]:
        # Custom aggregation logic
        pass
```

### Custom Partitioner

Implement the `Partitioner` base class:

```python
from unbitrium.partitioning.base import Partitioner

class CustomPartitioner(Partitioner):
    def partition(self, labels: np.ndarray) -> dict[int, list[int]]:
        # Custom partitioning logic
        pass
```

### Custom Metric

Create a pure function:

```python
def compute_custom_metric(
    labels: np.ndarray,
    client_indices: dict[int, list[int]],
) -> float:
    # Custom metric computation
    pass
```

---

## Dependencies

### Core Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| torch | >= 2.0 | Deep learning |
| numpy | >= 2.0 | Numerical computing |
| scipy | >= 1.12 | Scientific computing |
| pyyaml | >= 6.0 | Configuration |
| pydantic | >= 2.0 | Data validation |

### Optional Dependencies

| Package | Purpose |
|---------|---------|
| torchvision | Image datasets |
| matplotlib | Visualization |
| pandas | Data analysis |
| scikit-learn | ML utilities |

---

## Performance Considerations

### Vectorization

All metric computations use vectorized NumPy/PyTorch operations:

```python
# Vectorized EMD computation
distributions = np.array([...])  # (num_clients, num_classes)
global_dist = distributions.mean(axis=0)
emd_per_client = wasserstein_distance_batch(distributions, global_dist)
```

### GPU Acceleration

Model training and aggregation support GPU:

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
```

### Memory Efficiency

Large update collections are processed incrementally:

```python
# Streaming aggregation for memory efficiency
running_sum = None
for update in updates:
    if running_sum is None:
        running_sum = update["state_dict"]
    else:
        # Incremental aggregation
        ...
```

---

*Last updated: January 2026*
