"""Metrics module for evaluation framework."""

from typing import TYPE_CHECKING

from lightspeed_evaluation.core.system.lazy_import import create_lazy_getattr

if TYPE_CHECKING:
    # ruff: noqa: F401
    from lightspeed_evaluation.core.metrics.custom import CustomMetrics
    from lightspeed_evaluation.core.metrics.deepeval import DeepEvalMetrics
    from lightspeed_evaluation.core.metrics.nlp import NLPMetrics
    from lightspeed_evaluation.core.metrics.ragas import RagasMetrics
    from lightspeed_evaluation.core.metrics.script import ScriptEvalMetrics

_LAZY_IMPORTS = {
    "RagasMetrics": ("lightspeed_evaluation.core.metrics.ragas", "RagasMetrics"),
    "DeepEvalMetrics": (
        "lightspeed_evaluation.core.metrics.deepeval",
        "DeepEvalMetrics",
    ),
    "CustomMetrics": ("lightspeed_evaluation.core.metrics.custom", "CustomMetrics"),
    "ScriptEvalMetrics": (
        "lightspeed_evaluation.core.metrics.script",
        "ScriptEvalMetrics",
    ),
    "NLPMetrics": ("lightspeed_evaluation.core.metrics.nlp", "NLPMetrics"),
}

__getattr__ = create_lazy_getattr(_LAZY_IMPORTS, __name__)
