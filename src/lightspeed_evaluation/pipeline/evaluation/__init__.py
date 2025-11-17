"""Pipeline for the evaluation flow.

Uses lazy imports to defer loading evaluation components until needed.
"""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from lightspeed_evaluation.pipeline.evaluation.amender import APIDataAmender
    from lightspeed_evaluation.pipeline.evaluation.errors import EvaluationErrorHandler
    from lightspeed_evaluation.pipeline.evaluation.evaluator import MetricsEvaluator
    from lightspeed_evaluation.pipeline.evaluation.pipeline import EvaluationPipeline
    from lightspeed_evaluation.pipeline.evaluation.processor import (
        ConversationProcessor,
    )

__all__ = [
    "EvaluationPipeline",
    "APIDataAmender",
    "ConversationProcessor",
    "EvaluationErrorHandler",
    "MetricsEvaluator",
]


def __getattr__(name: str) -> Any:
    """Lazy import mechanism for pipeline evaluation module."""
    # pylint: disable=import-outside-toplevel
    if name == "EvaluationPipeline":
        from lightspeed_evaluation.pipeline.evaluation.pipeline import (
            EvaluationPipeline,
        )

        return EvaluationPipeline
    if name == "APIDataAmender":
        from lightspeed_evaluation.pipeline.evaluation.amender import APIDataAmender

        return APIDataAmender
    if name == "ConversationProcessor":
        from lightspeed_evaluation.pipeline.evaluation.processor import (
            ConversationProcessor,
        )

        return ConversationProcessor
    if name == "EvaluationErrorHandler":
        from lightspeed_evaluation.pipeline.evaluation.errors import (
            EvaluationErrorHandler,
        )

        return EvaluationErrorHandler
    if name == "MetricsEvaluator":
        from lightspeed_evaluation.pipeline.evaluation.evaluator import MetricsEvaluator

        return MetricsEvaluator

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
