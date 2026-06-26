"""Configuration of the project's logging system.

Configures a logger with a consistent format for use across all modules.
"""

import logging
import sys


def setup_logger(
    name: str = "flight_delay",
    level: int = logging.INFO,
) -> logging.Logger:
    """Create and configure a logger with console output.

    Args:
        name: Logger name.
        level: Logging level (DEBUG, INFO, WARNING, ERROR).

    Returns:
        Configured logger.
    """
    logger = logging.getLogger(name)

    # Avoid duplicating handlers if called multiple times
    if logger.handlers:
        return logger

    logger.setLevel(level)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger
