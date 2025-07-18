"""Custom exceptions for agent evaluation."""


class AgentEvaluationError(Exception):
    """Base exception for agent evaluation errors."""


class ConfigurationError(AgentEvaluationError):
    """Configuration-related errors."""


class AgentAPIError(AgentEvaluationError):
    """Agent API communication errors."""


class ScriptExecutionError(AgentEvaluationError):
    """Script execution errors."""


class JudgeModelError(AgentEvaluationError):
    """Judge model errors."""
