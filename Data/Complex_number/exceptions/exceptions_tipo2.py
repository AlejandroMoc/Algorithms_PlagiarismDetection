class EpsilonDeltaException(Exception):
    """Base exception class for EpsilonDelta operations."""
    pass

class InfinityError(EpsilonDeltaException):
    """Exception raised when attempting to divide by infinity."""
    pass