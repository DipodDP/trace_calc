import logging
import sys

def setup_logging(level=logging.INFO):
    """Set up logging configuration."""
    logging.basicConfig(
        level=level,
        format="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
        stream=sys.stdout,
    )

def get_logger(name):
    """Get a logger instance."""
    return logging.getLogger(name)
