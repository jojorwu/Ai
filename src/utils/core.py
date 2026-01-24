"""
Utility functions for the Transformer application.
"""
import logging


def setup_logging(log_path: str = None):
    """
    Configures logging to file and console.
    If log_path is None, only logs to console.
    """
    handlers = [logging.StreamHandler()]
    if log_path:
        handlers.append(logging.FileHandler(log_path))
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=handlers
    )
    logging.getLogger().setLevel(logging.INFO)


