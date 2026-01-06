"""Unbitrium: Federated Learning Heterogeneity Benchmarking Library.

A modular, research-grade Python library for quantifying and synthesizing
data heterogeneity in federated learning systems.

Author: Olaf Yunus Laitinen Imanov <oyli@dtu.dk>
License: EUPL-1.2
"""

from __future__ import annotations

__version__ = "1.0.0"
__author__ = "Olaf Yunus Laitinen Imanov"
__email__ = "oyli@dtu.dk"
__license__ = "EUPL-1.2"

# Aggregators
from unbitrium.aggregators import (
    AFL_DCS,
    Aggregator,
    FedAdam,
    FedAvg,
    FedCM,
    FedDyn,
    FedProx,
    FedSim,
    Krum,
    PFedSim,
    TrimmedMean,
)

# Benchmark
from unbitrium.bench import BenchmarkConfig, BenchmarkRunner

# Core imports
from unbitrium.core import (
    get_provenance_info,
    set_global_seed,
    set_seed,
    setup_logging,
)

# Datasets
from unbitrium.datasets import (
    DatasetRegistry,
    get_dataset,
    register_dataset,
)

# Metrics
from unbitrium.metrics import (
    compute_distribution_metrics,
    compute_drift_norm,
    compute_emd,
    compute_fairness_metrics,
    compute_gradient_variance,
    compute_imbalance_ratio,
    compute_js_divergence,
    compute_label_entropy,
    compute_nmi,
    compute_privacy_metrics,
)

# Partitioning
from unbitrium.partitioning import (
    DirichletPartitioner,
    EntropyControlledPartitioner,
    FeatureShiftPartitioner,
    MoDMPartitioner,
    Partitioner,
    QuantitySkewPartitioner,
)

# Privacy
from unbitrium.privacy import (
    GaussianMechanism,
    LaplaceMechanism,
    clip_gradients,
)

# Systems
from unbitrium.systems import Device, EnergyModel

__all__ = [
    # Version info
    "__version__",
    "__author__",
    "__email__",
    "__license__",
    # Core
    "setup_logging",
    "set_seed",
    "set_global_seed",
    "get_provenance_info",
    # Aggregators
    "Aggregator",
    "FedAvg",
    "FedProx",
    "FedSim",
    "PFedSim",
    "FedDyn",
    "FedCM",
    "FedAdam",
    "Krum",
    "TrimmedMean",
    "AFL_DCS",
    # Partitioning
    "Partitioner",
    "DirichletPartitioner",
    "MoDMPartitioner",
    "QuantitySkewPartitioner",
    "EntropyControlledPartitioner",
    "FeatureShiftPartitioner",
    # Metrics
    "compute_gradient_variance",
    "compute_drift_norm",
    "compute_imbalance_ratio",
    "compute_label_entropy",
    "compute_emd",
    "compute_js_divergence",
    "compute_nmi",
    "compute_distribution_metrics",
    "compute_fairness_metrics",
    "compute_privacy_metrics",
    # Datasets
    "DatasetRegistry",
    "register_dataset",
    "get_dataset",
    # Systems
    "Device",
    "EnergyModel",
    # Privacy
    "GaussianMechanism",
    "LaplaceMechanism",
    "clip_gradients",
    # Benchmark
    "BenchmarkRunner",
    "BenchmarkConfig",
]
