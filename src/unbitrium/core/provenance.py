"""Provenance Tracking for Reproducible Federated Learning Experiments.

Provides infrastructure for capturing and storing experiment metadata
to ensure reproducibility and auditability of federated learning runs.

Author: Olaf Yunus Laitinen Imanov <oyli@dtu.dk>
License: EUPL-1.2
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
from datetime import datetime
from typing import Any, Dict

from pydantic import BaseModel, Field


class ExperimentManifest(BaseModel):
    """Machine-readable manifest for experiment provenance.

    Captures all information needed to reproduce an experiment,
    including environment, configuration, and random seeds.

    Attributes:
        experiment_id: Unique identifier for the experiment.
        timestamp: ISO format timestamp of experiment start.
        python_version: Python interpreter version.
        platform: Operating system and version.
        hardware_info: Dictionary of hardware specifications.
        git_commit: Git commit hash of the codebase.
        git_dirty: Whether working directory had uncommitted changes.
        config: Complete experiment configuration.
        global_seed: Global random seed used.
        partition_seed: Seed used for data partitioning.
    """

    experiment_id: str
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    python_version: str = platform.python_version()
    platform: str = platform.platform()
    hardware_info: Dict[str, str] = Field(default_factory=dict)
    git_commit: str = ""
    git_dirty: bool = False
    config: Dict[str, Any] = Field(default_factory=dict)
    global_seed: int
    partition_seed: int


class ProvenanceTracker:
    """Captures and stores provenance information for experiments.

    Automatically collects environment, hardware, and version control
    information to create reproducibility manifests.

    Args:
        output_dir: Directory for storing provenance artifacts.

    Example:
        >>> tracker = ProvenanceTracker("./results")
        >>> manifest_path = tracker.save_manifest(config, seed=42)
    """

    def __init__(self, output_dir: str) -> None:
        """Initialize provenance tracker.

        Args:
            output_dir: Output directory for provenance files.
        """
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def capture_environment(self) -> Dict[str, str]:
        """Capture current hardware and system information.

        Returns:
            Dictionary containing processor, machine, OS, and
            framework version information.
        """
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
            info["torch_version"] = "not installed"

        return info

    def capture_git_info(self) -> Dict[str, Any]:
        """Capture current Git repository state.

        Returns:
            Dictionary with commit hash and dirty state.
        """
        try:
            commit = (
                subprocess.check_output(
                    ["git", "rev-parse", "HEAD"],
                    stderr=subprocess.DEVNULL,
                )
                .strip()
                .decode("utf-8")
            )
            status = (
                subprocess.check_output(
                    ["git", "status", "--porcelain"],
                    stderr=subprocess.DEVNULL,
                )
                .strip()
                .decode("utf-8")
            )
            return {"commit": commit, "dirty": bool(status)}
        except (subprocess.CalledProcessError, FileNotFoundError):
            return {"commit": "unknown", "dirty": False}

    def save_manifest(self, config: Dict[str, Any], seed: int) -> str:
        """Create and save the experiment manifest.

        Args:
            config: Experiment configuration dictionary.
            seed: Global random seed used for the experiment.

        Returns:
            Path to the saved manifest JSON file.
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

    def load_manifest(self, path: str) -> ExperimentManifest:
        """Load an experiment manifest from file.

        Args:
            path: Path to manifest JSON file.

        Returns:
            Loaded ExperimentManifest instance.
        """
        with open(path) as f:
            data = json.load(f)
        return ExperimentManifest(**data)
