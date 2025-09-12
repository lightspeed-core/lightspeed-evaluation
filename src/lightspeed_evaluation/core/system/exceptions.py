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
