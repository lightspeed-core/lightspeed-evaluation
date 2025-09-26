"""Exception definitions."""


class EvaluationError(Exception):
    """Base exception for evaluation framework."""


class APIError(EvaluationError):
    """Base exception for API-related errors."""


class ConfigurationError(EvaluationError):
    """Exception for configuration-related errors."""


class DataValidationError(EvaluationError):
    """Exception for data validation errors."""


class MetricError(EvaluationError):
    """Exception for metric evaluation errors."""


class LLMError(EvaluationError):
    """Exception for LLM-related errors."""


class ScriptExecutionError(EvaluationError):
    """Exception raised when script execution fails."""

    def __init__(self, message: str, script_path: str = "", return_code: int = -1):
        """Initialize script execution error.

        Args:
            message: Error message
            script_path: Path to the script that failed
            return_code: Return code from script execution
        """
        super().__init__(message)
        self.script_path = script_path
        self.return_code = return_code
