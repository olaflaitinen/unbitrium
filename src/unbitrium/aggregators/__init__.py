"""Aggregators package for Unbitrium.

Provides federated learning aggregation algorithms including FedAvg,
FedProx, FedSim, and robust aggregation methods.

Author: Olaf Yunus Laitinen Imanov <oyli@dtu.dk>
License: EUPL-1.2
"""

from __future__ import annotations

from unbitrium.aggregators.afl_dcs import AFL_DCS
from unbitrium.aggregators.base import Aggregator
from unbitrium.aggregators.fedadam import FedAdam
from unbitrium.aggregators.fedavg import FedAvg
from unbitrium.aggregators.fedcm import FedCM
from unbitrium.aggregators.feddyn import FedDyn
from unbitrium.aggregators.fedprox import FedProx
from unbitrium.aggregators.fedsim import FedSim
from unbitrium.aggregators.krum import Krum
from unbitrium.aggregators.pfedsim import PFedSim
from unbitrium.aggregators.trimmed_mean import TrimmedMean

__all__ = [
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
]
