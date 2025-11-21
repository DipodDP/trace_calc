# trace_calc/infrastructure/api/__init__.py
from .clients import AsyncElevationsApiClient, AsyncMagDeclinationApiClient

__all__ = [
    "AsyncElevationsApiClient", "AsyncMagDeclinationApiClient",
]
