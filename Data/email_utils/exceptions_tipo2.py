class XyzError(Exception):
    """Base exception for xyz operations."""
    pass

class AbcConnectionError(XyzError):
    """Raised when ABC connection fails."""
    pass

class DefError(XyzError):
    """Raised when handling DEF fails."""
    pass

class GhiError(XyzError):
    """Raised when template processing fails."""
    pass