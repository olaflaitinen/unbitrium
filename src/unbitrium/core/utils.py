"""Core utility functions for Unbitrium.

Provides logging setup, deterministic seeding, and provenance tracking
for reproducible federated learning experiments.

Author: Olaf Yunus Laitinen Imanov <oyli@dtu.dk>
License: EUPL-1.2
"""

from __future__ import annotations

import logging
import os
import platform
import random
import subprocess
import sys
from typing import Any

import numpy as np
import torch


def setup_logging(
    level: int = logging.INFO,
    name: str = "unbitrium",
) -> logging.Logger:
    """Configure logging for Unbitrium.

    Args:
        level: Logging level (default: INFO).
        name: Logger name (default: 'unbitrium').

    Returns:
        Configured logger instance.
    """
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=level,
    )
    logger = logging.getLogger(name)
    logger.setLevel(level)
    return logger


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility.

    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass


def set_global_seed(seed: int) -> None:
    """Alias for set_seed with additional environment configuration.

    Args:
        seed: Random seed value.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    set_seed(seed)


def get_provenance_info() -> dict[str, Any]:
    """Collect provenance information for experiment tracking.

    Returns:
        Dictionary containing git commit, Python version, library versions,
        hardware info, and environment details.
    """
    from datetime import datetime

    info: dict[str, Any] = {
        "timestamp": datetime.utcnow().isoformat(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "hostname": platform.node(),
        "machine": platform.machine(),
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "cuda_available": torch.cuda.is_available(),
    }

    if torch.cuda.is_available():
        info["cuda_version"] = torch.version.cuda
        info["gpu_name"] = torch.cuda.get_device_name(0)

    # Git commit
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            info["git_commit"] = result.stdout.strip()

        # Check for dirty state
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            info["git_dirty"] = len(result.stdout.strip()) > 0
    except Exception:
        info["git_commit"] = "unavailable"

    return info
