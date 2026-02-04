"""
Centralized logging configuration for the Titans project.
"""
import logging
import sys
from typing import Optional

def setup_logging(
    level: int = logging.INFO,
    log_path: Optional[str] = None,
    include_timestamp: bool = True
) -> None:
    """
    Configures the global logging settings.

    Args:
        level: Logging level (e.g., logging.INFO).
        log_path: Optional path to a file to save logs to.
        include_timestamp: Whether to include timestamps in logs.
    """
    handlers = [logging.StreamHandler(sys.stdout)]

    if log_path:
        handlers.append(logging.FileHandler(log_path))

    fmt = '%(asctime)s [%(levelname)s] %(name)s: %(message)s' if include_timestamp \
          else '[%(levelname)s] %(name)s: %(message)s'

    # Remove any existing handlers
    root = logging.getLogger()
    if root.handlers:
        for handler in root.handlers:
            root.removeHandler(handler)

    logging.basicConfig(
        level=level,
        format=fmt,
        handlers=handlers,
        force=True
    )

    # Optional: Suppress noisy third-party loggers
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("accelerate").setLevel(logging.INFO)
