class AnalyzerException(Exception):
    """
    Base exception for all analyzer-related errors.
    """


class CoordinatesRequiredException(AnalyzerException):
    """
    Raised when required site coordinates are missing.
    """


class APIException(AnalyzerException):
    """
    Base exception for API-related errors.
    """


class RateLimitException(APIException):
    """
    Raised when API rate limit is exceeded (HTTP 429).
    Safe to retry with exponential backoff.
    """


class AuthenticationException(APIException):
    """
    Raised when API authentication fails (HTTP 401/403).
    Do NOT retry - requires user intervention.
    """


class TransientAPIException(APIException):
    """
    Raised for temporary API failures (HTTP 503, timeouts, network errors).
    Safe to retry with exponential backoff.
    """


class InvalidResponseException(APIException):
    """
    Raised when API returns malformed or unexpected data.
    Do NOT retry - indicates data quality issue.
    """
