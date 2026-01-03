"""
Structured logging configuration.
"""

import logging
import sys
from typing import Optional

def configure_logging(
    level: int = logging.INFO,
    log_file: Optional[str] = None
) -> None:
    """
    Configures the root logger for Unbitrium.

    Parameters
    ----------
    level : int
        Logging level (e.g., logging.INFO).
    log_file : Optional[str]
        Path to a log file. If None, logs only to stdout.
    """
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        handlers=handlers,
        force=True
    )
