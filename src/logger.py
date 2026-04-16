"""Structured logging configuration for the agent pipeline."""

import logging
import sys
from typing import Any, Dict, Optional

from src.config import settings


def _build_logger(name: str) -> logging.Logger:
    """Create and configure a named logger.

    Args:
        name: Logger namespace, typically the module's ``__name__``.

    Returns:
        Configured :class:`logging.Logger` instance.
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger  # already configured in this process

    level = getattr(logging, settings.log_level.upper(), logging.INFO)
    logger.setLevel(level)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False
    return logger


# Module-level default logger; other modules import this.
log = _build_logger("agent")


def log_tool_call(
    tool_name: str,
    parameters: Dict[str, Any],
    result_type: str,
    duration_ms: float,
    attempt_count: int,
    status: str,
    error: Optional[str] = None,
) -> None:
    """Emit a structured log line for a completed tool call.

    Args:
        tool_name: Name of the tool that was executed.
        parameters: Input parameters passed to the tool.
        result_type: Python type name of the return value (or "N/A" on error).
        duration_ms: Wall-clock execution time in milliseconds.
        attempt_count: How many attempts were made (≥1).
        status: "success" | "error" | "skipped".
        error: Error message string if status is "error", otherwise ``None``.
    """
    if not settings.log_tool_calls:
        return

    base = (
        f"[TOOL:{status.upper()}] {tool_name} | "
        f"params={parameters} | "
        f"result_type={result_type} | "
        f"duration={duration_ms:.1f}ms | "
        f"attempts={attempt_count}"
    )
    if error:
        base += f" | error={error}"

    if status == "success":
        log.info(base)
    elif status == "error":
        log.warning(base)
    else:
        log.debug(base)
