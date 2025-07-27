"""Custom exceptions for agent evaluation."""


class AgentEvaluationError(Exception):
    """Base exception for agent evaluation errors."""


class EvaluationDataError(AgentEvaluationError):
    """Evaluation data loading, parsing, and validation errors."""


class AgentAPIError(AgentEvaluationError):
    """Agent API communication errors."""


class ScriptExecutionError(AgentEvaluationError):
    """Script execution errors."""


class JudgeModelError(AgentEvaluationError):
    """Judge model errors."""
