"""LLM management for Evaluation Framework.

Uses lazy imports to defer loading heavy framework-specific LLM implementations.
"""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from lightspeed_evaluation.core.llm.custom import BaseCustomLLM
    from lightspeed_evaluation.core.llm.deepeval import DeepEvalLLMManager
    from lightspeed_evaluation.core.llm.manager import LLMManager
    from lightspeed_evaluation.core.llm.ragas import RagasLLMManager
    from lightspeed_evaluation.core.models import LLMConfig
    from lightspeed_evaluation.core.system.env_validator import validate_provider_env
    from lightspeed_evaluation.core.system.exceptions import LLMError

__all__ = [
    "LLMConfig",
    "LLMError",
    "LLMManager",
    "BaseCustomLLM",
    "DeepEvalLLMManager",
    "RagasLLMManager",
    "validate_provider_env",
]


def __getattr__(name: str) -> Any:  # pylint: disable=too-many-return-statements
    """Lazy import mechanism for LLM module.

    Framework-specific LLM managers are only imported when actually used.
    """
    # pylint: disable=import-outside-toplevel
    if name == "LLMManager":
        from lightspeed_evaluation.core.llm.manager import LLMManager

        return LLMManager
    if name == "BaseCustomLLM":
        from lightspeed_evaluation.core.llm.custom import BaseCustomLLM

        return BaseCustomLLM
    if name == "DeepEvalLLMManager":
        from lightspeed_evaluation.core.llm.deepeval import DeepEvalLLMManager

        return DeepEvalLLMManager
    if name == "RagasLLMManager":
        from lightspeed_evaluation.core.llm.ragas import RagasLLMManager

        return RagasLLMManager
    if name == "LLMConfig":
        from lightspeed_evaluation.core.models import LLMConfig

        return LLMConfig
    if name == "validate_provider_env":
        from lightspeed_evaluation.core.system.env_validator import (
            validate_provider_env,
        )

        return validate_provider_env
    if name == "LLMError":
        from lightspeed_evaluation.core.system.exceptions import LLMError

        return LLMError

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
