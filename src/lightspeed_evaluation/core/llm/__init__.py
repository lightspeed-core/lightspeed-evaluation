"""LLM management for Evaluation Framework."""

from typing import TYPE_CHECKING

from lightspeed_evaluation.core.system.lazy_import import create_lazy_getattr

if TYPE_CHECKING:
    # ruff: noqa: F401
    from lightspeed_evaluation.core.llm.custom import BaseCustomLLM, TokenTracker
    from lightspeed_evaluation.core.llm.deepeval import DeepEvalLLMManager
    from lightspeed_evaluation.core.llm.manager import LLMManager
    from lightspeed_evaluation.core.llm.ragas import RagasLLMManager
    from lightspeed_evaluation.core.models import LLMConfig
    from lightspeed_evaluation.core.system.env_validator import validate_provider_env
    from lightspeed_evaluation.core.system.exceptions import LLMError

_LAZY_IMPORTS = {
    "LLMConfig": ("lightspeed_evaluation.core.models", "LLMConfig"),
    "LLMError": ("lightspeed_evaluation.core.system.exceptions", "LLMError"),
    "LLMManager": ("lightspeed_evaluation.core.llm.manager", "LLMManager"),
    "BaseCustomLLM": ("lightspeed_evaluation.core.llm.custom", "BaseCustomLLM"),
    "TokenTracker": ("lightspeed_evaluation.core.llm.custom", "TokenTracker"),
    "DeepEvalLLMManager": (
        "lightspeed_evaluation.core.llm.deepeval",
        "DeepEvalLLMManager",
    ),
    "RagasLLMManager": ("lightspeed_evaluation.core.llm.ragas", "RagasLLMManager"),
    "validate_provider_env": (
        "lightspeed_evaluation.core.system.env_validator",
        "validate_provider_env",
    ),
}

__getattr__ = create_lazy_getattr(_LAZY_IMPORTS, __name__)
__all__ = list(_LAZY_IMPORTS.keys())
