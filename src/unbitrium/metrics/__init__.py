"""Metrics package for Unbitrium.

Provides heterogeneity, fairness, distribution, and other federated
learning evaluation metrics.

Author: Olaf Yunus Laitinen Imanov <oyli@dtu.dk>
License: EUPL-1.2
"""

from __future__ import annotations

from unbitrium.metrics.heterogeneity import (
    compute_gradient_variance,
    compute_drift_norm,
    compute_imbalance_ratio,
    compute_label_entropy,
    compute_emd,
    compute_js_divergence,
    compute_nmi,
)
from unbitrium.metrics.distribution import compute_distribution_metrics
from unbitrium.metrics.fairness import compute_fairness_metrics
from unbitrium.metrics.privacy import compute_privacy_metrics

__all__ = [
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
]
