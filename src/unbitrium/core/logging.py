"""Structured Logging Configuration for Unbitrium.

Provides standardized logging setup for federated learning
experiments with support for console and file output.

Author: Olaf Yunus Laitinen Imanov <oyli@dtu.dk>
License: EUPL-1.2
"""

from __future__ import annotations

import logging
import sys
from typing import Optional


def configure_logging(
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    format_string: Optional[str] = None,
) -> None:
    """Configure the root logger for Unbitrium.

    Sets up logging with a standardized format for console output
    and optional file logging.

    Args:
        level: Logging level (e.g., logging.INFO, logging.DEBUG).
        log_file: Path to log file. If None, logs only to console.
        format_string: Custom format string. Uses default if None.

    Example:
        >>> configure_logging(level=logging.DEBUG, log_file="./run.log")
    """
    if format_string is None:
        format_string = "%(asctime)s | %(name)s | %(levelname)s | %(message)s"

    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=level,
        format=format_string,
        handlers=handlers,
        force=True,
    )


def get_logger(name: str, level: Optional[int] = None) -> logging.Logger:
    """Get a named logger for a module.

    Args:
        name: Logger name (typically __name__).
        level: Optional override for logging level.

    Returns:
        Configured Logger instance.

    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("Training started")
    """
    logger = logging.getLogger(name)
    if level is not None:
        logger.setLevel(level)
    return logger


class SimulationLogger:
    """Structured logger for simulation events.

    Provides consistent logging methods for common simulation
    events like round starts, client training, and aggregation.

    Args:
        name: Logger name.

    Example:
        >>> sim_log = SimulationLogger("experiment_001")
        >>> sim_log.round_start(round_num=5)
    """

    def __init__(self, name: str = "unbitrium.simulation") -> None:
        """Initialize simulation logger.

        Args:
            name: Logger name.
        """
        self.logger = logging.getLogger(name)

    def round_start(self, round_num: int) -> None:
        """Log round start."""
        self.logger.info(f"Round {round_num} started")

    def round_end(self, round_num: int, metrics: dict) -> None:
        """Log round end with metrics."""
        self.logger.info(f"Round {round_num} completed: {metrics}")

    def client_selected(self, client_ids: list[int]) -> None:
        """Log client selection."""
        self.logger.debug(f"Selected clients: {client_ids}")

    def aggregation_complete(self, num_clients: int) -> None:
        """Log aggregation completion."""
        self.logger.info(f"Aggregation complete with {num_clients} clients")

    def error(self, message: str, exc_info: bool = True) -> None:
        """Log error with optional exception info."""
        self.logger.error(message, exc_info=exc_info)
