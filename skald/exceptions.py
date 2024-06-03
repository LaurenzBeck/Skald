"""Module containing custom Exceptions 🧨💥."""


class MissingIDsError(ValueError):
    """Raised when trying to log a metric without one or multiple IDs."""


class DuplicateIDsError(ValueError):
    """Raised when trying to log a metric that was already logged."""
