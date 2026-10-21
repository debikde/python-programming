"""Logging configuration helpers."""

from __future__ import annotations

import logging
import sys

NOISY_LOGGERS = ("httpx", "urllib3")


def setup_logging(level: str) -> None:
    """Configure root logging for CLI runs and tests."""

    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
        handler.close()

    numeric_level = getattr(logging, level.upper(), logging.INFO)
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s %(levelname)s %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )

    root_logger.addHandler(handler)
    root_logger.setLevel(numeric_level)

    for logger_name in NOISY_LOGGERS:
        logging.getLogger(logger_name).setLevel(logging.WARNING)
