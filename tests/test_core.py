"""Unit tests for core module.

Author: Olaf Yunus Laitinen Imanov <oyli@dtu.dk>
License: EUPL-1.2
"""

from __future__ import annotations

import logging
import pytest
import torch
import numpy as np

from unbitrium.core import (
    setup_logging,
    set_seed,
    set_global_seed,
    get_provenance_info,
)


class TestSeeding:
    """Tests for random seeding functions."""

    def test_set_seed_reproducibility(self) -> None:
        """Test that set_seed produces reproducible results."""
        set_seed(123)
        values1 = [np.random.rand(), torch.rand(1).item()]

        set_seed(123)
        values2 = [np.random.rand(), torch.rand(1).item()]

        assert values1[0] == pytest.approx(values2[0])
        assert values1[1] == pytest.approx(values2[1])

    def test_set_global_seed(self) -> None:
        """Test global seed setting."""
        set_global_seed(42)
        val1 = np.random.rand()

        set_global_seed(42)
        val2 = np.random.rand()

        assert val1 == pytest.approx(val2)

    def test_different_seeds_different_values(self) -> None:
        """Test different seeds produce different values."""
        set_seed(100)
        val1 = np.random.rand()

        set_seed(200)
        val2 = np.random.rand()

        assert val1 != val2


class TestLogging:
    """Tests for logging setup."""

    def test_setup_logging_default(self) -> None:
        """Test default logging setup."""
        logger = setup_logging()
        assert logger is not None
        assert isinstance(logger, logging.Logger)

    def test_setup_logging_custom_level(self) -> None:
        """Test custom log level."""
        logger = setup_logging(level=logging.DEBUG)
        assert logger.level == logging.DEBUG

    def test_setup_logging_with_name(self) -> None:
        """Test logging with custom name."""
        logger = setup_logging(name="test_logger")
        assert logger.name == "test_logger"


class TestProvenance:
    """Tests for provenance tracking."""

    def test_get_provenance_info(self) -> None:
        """Test provenance info retrieval."""
        info = get_provenance_info()

        assert isinstance(info, dict)
        assert "timestamp" in info
        assert "python_version" in info
        assert "torch_version" in info
        assert "numpy_version" in info

    def test_provenance_contains_system_info(self) -> None:
        """Test provenance contains system information."""
        info = get_provenance_info()

        assert "platform" in info
        assert "hostname" in info or "machine" in info

    def test_provenance_is_serializable(self) -> None:
        """Test provenance info can be serialized."""
        import json

        info = get_provenance_info()
        serialized = json.dumps(info)
        assert isinstance(serialized, str)
        assert len(serialized) > 0


class TestModuleExports:
    """Test that all expected items are exported from core."""

    def test_exports_exist(self) -> None:
        """Test core exports are available."""
        from unbitrium import core

        assert hasattr(core, "setup_logging")
        assert hasattr(core, "set_seed")
        assert hasattr(core, "set_global_seed")
        assert hasattr(core, "get_provenance_info")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
