"""Storage-related exception definitions.

This module defines exceptions for storage operations. All exceptions
inherit from the base EvaluationError for consistent error handling.
"""

from lightspeed_evaluation.core.system.exceptions import EvaluationError


class StorageError(EvaluationError):
    """Exception for storage-related errors.

    This exception is raised when storage operations fail, such as
    initialization failures, save errors, or finalization issues.

    Attributes:
        backend_name: Optional name of the backend that raised the error.
    """

    def __init__(self, message: str, backend_name: str | None = None):
        """Initialize storage error.

        Args:
            message: Human-readable error description.
            backend_name: Optional name of the backend that raised the error.
        """
        super().__init__(message)
        self.backend_name = backend_name
