"""Benchmark package for Unbitrium.

Provides standardized benchmark harness for federated learning experiments.

Author: Olaf Yunus Laitinen Imanov <oyli@dtu.dk>
License: EUPL-1.2
"""

from __future__ import annotations

from unbitrium.bench.config import BenchmarkConfig
from unbitrium.bench.runner import BenchmarkRunner

__all__ = [
    "BenchmarkRunner",
    "BenchmarkConfig",
]
