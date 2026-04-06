"""Logging setup for cave-talk."""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from .config import LOG_DIR, ensure_dirs


def setup_logging() -> logging.Logger:
    """Configure file-based logging. Returns the root cave-talk logger."""
    ensure_dirs()

    logger = logging.getLogger("cave_talk")
    if logger.handlers:
        return logger  # already configured

    logger.setLevel(logging.DEBUG)

    today = datetime.now(timezone.utc).strftime("%Y%m%d")
    log_path = LOG_DIR / f"{today}.log"

    handler = logging.FileHandler(log_path)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    ))

    logger.addHandler(handler)
    return logger
