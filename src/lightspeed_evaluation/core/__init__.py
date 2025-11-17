"""Core functionality - Components for evaluation.

Uses lazy imports to avoid loading heavy dependencies until needed.
"""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from lightspeed_evaluation.core.llm import LLMManager
    from lightspeed_evaluation.core.models import (
        EvaluationData,
        EvaluationResult,
        LLMConfig,
        TurnData,
    )
    from lightspeed_evaluation.core.system import (
        ConfigLoader,
        DataValidator,
        SystemConfig,
    )

__all__ = [
    # Configuration & data
    "ConfigLoader",
    "SystemConfig",
    "DataValidator",
    # Data models
    "EvaluationData",
    "EvaluationResult",
    "TurnData",
    "LLMConfig",
    # LLM management
    "LLMManager",
]


def __getattr__(name: str) -> Any:
    """Lazy import mechanism for core module imports."""
    # pylint: disable=import-outside-toplevel
    if name == "LLMManager":
        from lightspeed_evaluation.core.llm import LLMManager

        return LLMManager
    if name in ("ConfigLoader", "SystemConfig", "DataValidator"):
        from lightspeed_evaluation.core.system import (
            ConfigLoader,
            DataValidator,
            SystemConfig,
        )

        return {
            "ConfigLoader": ConfigLoader,
            "SystemConfig": SystemConfig,
            "DataValidator": DataValidator,
        }[name]
    if name in ("EvaluationData", "EvaluationResult", "TurnData", "LLMConfig"):
        from lightspeed_evaluation.core.models import (
            EvaluationData,
            EvaluationResult,
            LLMConfig,
            TurnData,
        )

        return {
            "EvaluationData": EvaluationData,
            "EvaluationResult": EvaluationResult,
            "TurnData": TurnData,
            "LLMConfig": LLMConfig,
        }[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
