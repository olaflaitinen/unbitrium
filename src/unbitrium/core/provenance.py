"""
Provenance tracking for reproducible experiments.
"""

import json
import os
import sys
import platform
import subprocess
from datetime import datetime
from typing import Any, Dict

from pydantic import BaseModel, Field

class ExperimentManifest(BaseModel):
    """Machine-readable manifest for experiment provenance."""

    experiment_id: str
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())

    # Environment
    python_version: str = platform.python_version()
    platform: str = platform.platform()
    hardware_info: Dict[str, str] = Field(default_factory=dict)

    # Git info
    git_commit: str = ""
    git_dirty: bool = False

    # Configuration
    config: Dict[str, Any] = Field(default_factory=dict)

    # Seeds
    global_seed: int
    partition_seed: int

class ProvenanceTracker:
    """
    Captures and stores provenance information for an experiment.
    """

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def capture_environment(self) -> Dict[str, str]:
        """Captures hardware and system info."""
        info = {
            "processor": platform.processor(),
            "machine": platform.machine(),
            "os_release": platform.release(),
        }
        try:
            import torch
            info["torch_version"] = torch.__version__
            info["cuda_available"] = str(torch.cuda.is_available())
            if torch.cuda.is_available():
                info["cuda_device"] = torch.cuda.get_device_name(0)
        except ImportError:
            pass

        return info

    def capture_git_info(self) -> Dict[str, Any]:
        """Captures current git commit and status."""
        try:
            commit = subprocess.check_output(["git", "rev-parse", "HEAD"]).strip().decode("utf-8")
            status = subprocess.check_output(["git", "status", "--porcelain"]).strip().decode("utf-8")
            return {
                "commit": commit,
                "dirty": bool(status),
            }
        except subprocess.CalledProcessError:
            return {"commit": "unknown", "dirty": False}

    def save_manifest(self, config: Dict[str, Any], seed: int) -> str:
        """
        Creates and saves the experiment manifest.

        Parameters
        ----------
        config : Dict[str, Any]
            The normalized configuration dictionary.
        seed : int
            The global random seed used.

        Returns
        -------
        str
            Path to the saved manifest file.
        """
        git_info = self.capture_git_info()
        env_info = self.capture_environment()

        manifest = ExperimentManifest(
            experiment_id=config.get("experiment", {}).get("name", "unnamed"),
            hardware_info=env_info,
            git_commit=git_info["commit"],
            git_dirty=git_info["dirty"],
            config=config,
            global_seed=seed,
            partition_seed=config.get("partitioning", {}).get("seed", seed),
        )

        path = os.path.join(self.output_dir, "manifest.json")
        with open(path, "w") as f:
            f.write(manifest.model_dump_json(indent=2))

        return path
