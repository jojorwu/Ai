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


def main_entrypoint(main_func):
    """
    Decorator to wrap main functions with common exception handling.
    """

    def wrapper():
        try:
            main_func()
        except FileNotFoundError as e:
            logging.error("File not found: %s", e)
        except (ValueError, TypeError) as e:
            logging.error("Configuration or value error: %s", e)
        except KeyboardInterrupt:
            logging.info("\nExecution interrupted by user.")
        except Exception as e:
            logging.error("An unexpected error occurred: %s", e, exc_info=True)
            raise

    return wrapper
