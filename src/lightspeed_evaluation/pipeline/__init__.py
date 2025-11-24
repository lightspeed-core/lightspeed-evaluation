"""Pipeline for evaluation framework."""

from typing import TYPE_CHECKING

from lightspeed_evaluation.core.system.lazy_import import create_lazy_getattr

if TYPE_CHECKING:
    # ruff: noqa: F401
    from lightspeed_evaluation.pipeline.evaluation import EvaluationPipeline

_LAZY_IMPORTS = {
    "EvaluationPipeline": (
        "lightspeed_evaluation.pipeline.evaluation",
        "EvaluationPipeline",
    ),
}

__getattr__ = create_lazy_getattr(_LAZY_IMPORTS, __name__)
__all__ = list(_LAZY_IMPORTS.keys())
