"""Pipeline for the evaluation flow."""

from typing import TYPE_CHECKING

from lightspeed_evaluation.core.system.lazy_import import create_lazy_getattr

if TYPE_CHECKING:
    # ruff: noqa: F401
    from lightspeed_evaluation.pipeline.evaluation.amender import APIDataAmender
    from lightspeed_evaluation.pipeline.evaluation.driver import (
        AgentDriver,
        ProposalDriver,
    )
    from lightspeed_evaluation.pipeline.evaluation.errors import EvaluationErrorHandler
    from lightspeed_evaluation.pipeline.evaluation.evaluator import MetricsEvaluator
    from lightspeed_evaluation.pipeline.evaluation.pipeline import EvaluationPipeline
    from lightspeed_evaluation.pipeline.evaluation.processor import (
        ConversationProcessor,
    )
    from lightspeed_evaluation.pipeline.evaluation.registry import AgentDriverRegistry

_LAZY_IMPORTS = {
    "EvaluationPipeline": (
        "lightspeed_evaluation.pipeline.evaluation.pipeline",
        "EvaluationPipeline",
    ),
    "APIDataAmender": (
        "lightspeed_evaluation.pipeline.evaluation.amender",
        "APIDataAmender",
    ),
    "AgentDriver": (
        "lightspeed_evaluation.pipeline.evaluation.driver",
        "AgentDriver",
    ),
    "AgentDriverRegistry": (
        "lightspeed_evaluation.pipeline.evaluation.registry",
        "AgentDriverRegistry",
    ),
    "ConversationProcessor": (
        "lightspeed_evaluation.pipeline.evaluation.processor",
        "ConversationProcessor",
    ),
    "EvaluationErrorHandler": (
        "lightspeed_evaluation.pipeline.evaluation.errors",
        "EvaluationErrorHandler",
    ),
    "MetricsEvaluator": (
        "lightspeed_evaluation.pipeline.evaluation.evaluator",
        "MetricsEvaluator",
    ),
    "ProposalDriver": (
        "lightspeed_evaluation.pipeline.evaluation.driver",
        "ProposalDriver",
    ),
}

__getattr__ = create_lazy_getattr(_LAZY_IMPORTS, __name__)
