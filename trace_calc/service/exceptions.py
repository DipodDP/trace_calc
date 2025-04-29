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
    Raised when API data cannot be retrieved.
    """
