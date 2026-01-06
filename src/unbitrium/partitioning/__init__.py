"""Partitioning package for Unbitrium.

Provides data partitioning strategies for creating non-IID federated
learning scenarios.

Author: Olaf Yunus Laitinen Imanov <oyli@dtu.dk>
License: EUPL-1.2
"""

from __future__ import annotations

from unbitrium.partitioning.base import Partitioner
from unbitrium.partitioning.dirichlet import DirichletLabelSkew, DirichletPartitioner
from unbitrium.partitioning.entropy_controlled import EntropyControlledPartitioner
from unbitrium.partitioning.feature_shift import FeatureShiftPartitioner
from unbitrium.partitioning.modm import MoDMPartitioner
from unbitrium.partitioning.quantity_skew import QuantitySkewPartitioner

__all__ = [
    "Partitioner",
    "DirichletPartitioner",
    "DirichletLabelSkew",
    "MoDMPartitioner",
    "QuantitySkewPartitioner",
    "EntropyControlledPartitioner",
    "FeatureShiftPartitioner",
]
