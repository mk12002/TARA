"""
Logging Configuration
========================
Structured logging using Loguru.
"""

import sys
from loguru import logger


def setup_logging(level: str = "INFO"):
    """Configure Loguru for TARA."""
    logger.remove()  # Remove default handler

    # Console handler with colors
    logger.add(
        sys.stderr,
        level=level,
        format=(
            "<green>{time:HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{module}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        ),
        colorize=True,
    )

    # File handler
    logger.add(
        "logs/tara_{time:YYYY-MM-DD}.log",
        level="DEBUG",
        rotation="10 MB",
        retention="7 days",
        compression="gz",
    )

    logger.info(f"Logging initialized at {level} level")
