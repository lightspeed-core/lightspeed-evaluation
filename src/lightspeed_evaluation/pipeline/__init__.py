"""Pipeline for evaluation framework."""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from lightspeed_evaluation.pipeline.evaluation import EvaluationPipeline

__all__ = [
    "EvaluationPipeline",
]


def __getattr__(name: str) -> Any:
    """Lazy import mechanism for pipeline module."""
    # pylint: disable=import-outside-toplevel
    if name == "EvaluationPipeline":
        from lightspeed_evaluation.pipeline.evaluation import EvaluationPipeline

        return EvaluationPipeline

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
