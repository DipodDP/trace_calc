import asyncio
import httpx
from functools import wraps
from trace_calc.logging_config import get_logger
from trace_calc.domain.exceptions import APIException

logger = get_logger(__name__)

def async_retry(max_retries=5, backoff_factor=0.5, initial_timeout=10.0, max_timeout=30.0):
    """
    Decorator for retrying async functions with exponential backoff.

    Handles network errors, timeouts, proxy errors, and API exceptions.
    Increases timeout on each retry to handle slow connections.
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries):
                try:
                    # Calculate timeout for the current attempt
                    timeout = min(initial_timeout * (2 ** attempt), max_timeout)
                    kwargs['timeout'] = timeout

                    logger.info(f"Attempt {attempt + 1}/{max_retries} with timeout {timeout:.1f}s for {func.__name__}")

                    return await func(*args, **kwargs)
                except (
                    APIException,
                    httpx.TimeoutException,
                    httpx.ConnectTimeout,
                    httpx.ReadTimeout,
                    httpx.WriteTimeout,
                    httpx.PoolTimeout,
                    httpx.ConnectError,
                    httpx.ProxyError,
                    httpx.NetworkError,
                    asyncio.TimeoutError,
                ) as e:
                    last_exception = e
                    error_type = type(e).__name__

                    if attempt < max_retries - 1:
                        delay = backoff_factor * (2 ** attempt)
                        logger.warning(
                            f"{error_type} in {func.__name__}: {e}. "
                            f"Retrying in {delay:.2f}s (attempt {attempt + 1}/{max_retries})..."
                        )
                        await asyncio.sleep(delay)
                    else:
                        logger.error(
                            f"All {max_retries} retry attempts for {func.__name__} failed. "
                            f"Last error: {error_type}: {e}"
                        )
                        raise e
            # This part should not be reachable if max_retries > 0
            if last_exception:
                raise last_exception
            return None
        return wrapper
    return decorator
