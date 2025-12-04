import logging
import sys
from environs import Env
from .log_filters import TruncatingFilter


def setup_logging(env: Env) -> None:
    """Set up logging configuration."""
    if logging.root.handlers:  # Check if logging is already configured
        return

    log_level_str = env.str("LOGGING_LEVEL", "INFO").upper()

    # DEBUG flag overrides log level when set to True
    debug_mode = env.bool("DEBUG", default=False)
    if debug_mode:
        log_level_str = "DEBUG"

    numeric_level = getattr(logging, log_level_str, None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {log_level_str}")

    logging.basicConfig(
        level=numeric_level,
        format="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
        stream=sys.stdout,
    )

    # Set the level for the httpx logger, add the truncating filter, and disable propagation
    httpx_logger = logging.getLogger("httpx")
    httpx_logger.setLevel(numeric_level)
    httpx_logger.addFilter(TruncatingFilter(max_length=105))
    httpx_logger.propagate = False

    # Set the level for the httpcore logger
    logging.getLogger("httpcore").setLevel(numeric_level)

    # Set the level for our custom API client logger
    api_clients_logger = logging.getLogger("trace_calc.infrastructure.api.clients")
    api_clients_logger.setLevel(numeric_level)
    api_clients_logger.addFilter(TruncatingFilter(max_length=105))


    # Set the level for the matplotlib logger to INFO to reduce verbosity
    logging.getLogger("matplotlib").setLevel(logging.INFO)
    logging.getLogger("matplotlib.font_manager").setLevel(logging.INFO)
    logging.getLogger("matplotlib.pyplot").setLevel(logging.INFO)

    # Additionally, force the level on all *existing* loggers
    # This catches loggers that might have been created before setup_logging ran
    for logger_name in logging.Logger.manager.loggerDict:
        logger = logging.getLogger(logger_name)
        if logger.level == logging.NOTSET:  # Only set if not explicitly set already
            logger.setLevel(numeric_level)


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance."""
    return logging.getLogger(name)
