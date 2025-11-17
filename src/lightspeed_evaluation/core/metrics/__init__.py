"""Metrics module for evaluation framework.

Uses lazy imports to defer loading heavy dependencies (ragas, deepeval)
until specific metric frameworks are actually used.
"""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from lightspeed_evaluation.core.metrics.custom import CustomMetrics
    from lightspeed_evaluation.core.metrics.deepeval import DeepEvalMetrics
    from lightspeed_evaluation.core.metrics.ragas import RagasMetrics
    from lightspeed_evaluation.core.metrics.script import ScriptEvalMetrics

__all__ = ["RagasMetrics", "DeepEvalMetrics", "CustomMetrics", "ScriptEvalMetrics"]


def __getattr__(name: str) -> Any:
    """Lazy import mechanism for metrics.

    This ensures that heavy frameworks like ragas and deepeval are only
    imported when their respective metric classes are actually instantiated.
    """
    # pylint: disable=import-outside-toplevel
    if name == "RagasMetrics":
        from lightspeed_evaluation.core.metrics.ragas import RagasMetrics

        return RagasMetrics
    if name == "DeepEvalMetrics":
        from lightspeed_evaluation.core.metrics.deepeval import DeepEvalMetrics

        return DeepEvalMetrics
    if name == "CustomMetrics":
        from lightspeed_evaluation.core.metrics.custom import CustomMetrics

        return CustomMetrics
    if name == "ScriptEvalMetrics":
        from lightspeed_evaluation.core.metrics.script import ScriptEvalMetrics

        return ScriptEvalMetrics

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
