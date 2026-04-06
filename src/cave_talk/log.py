"""Logging setup for cave-talk."""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone

from .config import LOG_DIR, ensure_dirs


def _resolve_level() -> int:
    """Return the log level from env var or default to INFO."""
    env = os.environ.get("CAVE_TALK_LOG", "").strip().upper()
    if env:
        return getattr(logging, env, logging.INFO)
    return logging.INFO


def setup_logging() -> logging.Logger:
    """Configure file-based logging. Returns the root cave-talk logger.

    Set ``CAVE_TALK_LOG=DEBUG`` to enable debug-level output.
    """
    ensure_dirs()

    logger = logging.getLogger("cave_talk")
    if logger.handlers:
        return logger  # already configured

    level = _resolve_level()
    logger.setLevel(level)

    today = datetime.now(timezone.utc).strftime("%Y%m%d")
    log_path = LOG_DIR / f"{today}.log"

    handler = logging.FileHandler(log_path)
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    ))

    logger.addHandler(handler)

    # Also log to stderr so crashes are visible in the terminal
    stderr_handler = logging.StreamHandler()
    stderr_handler.setLevel(level)
    stderr_handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    ))
    logger.addHandler(stderr_handler)

    return logger
