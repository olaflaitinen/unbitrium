"""
Federated Learning Aggregation Algorithms.
"""

from unbitrium.aggregators.base import Aggregator
from unbitrium.aggregators.fedavg import FedAvg
from unbitrium.aggregators.fedprox import FedProx
from unbitrium.aggregators.feddyn import FedDyn
from unbitrium.aggregators.fedsim import FedSim
from unbitrium.aggregators.pfedsim import pFedSim
from unbitrium.aggregators.fedcm import FedCM
from unbitrium.aggregators.afl_dcs import AFL_DCS
from unbitrium.aggregators.fedopt import FedOpt, FedAdam, FedYogi, FedAdagrad
from unbitrium.aggregators.robust import TrimmedMean, Krum

__all__ = [
    "Aggregator",
    "FedAvg",
    "FedProx",
    "FedDyn",
    "FedSim",
    "pFedSim",
    "FedCM",
    "AFL_DCS",
    "FedOpt",
    "FedAdam",
    "FedYogi",
    "FedAdagrad",
    "TrimmedMean",
    "Krum",
]
