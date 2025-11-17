"""Script execution module for evaluation framework."""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from lightspeed_evaluation.core.script.manager import ScriptExecutionManager
    from lightspeed_evaluation.core.system.exceptions import ScriptExecutionError

__all__ = [
    "ScriptExecutionManager",
    "ScriptExecutionError",
]


def __getattr__(name: str) -> Any:
    """Lazy import mechanism for script module."""
    # pylint: disable=import-outside-toplevel
    if name == "ScriptExecutionManager":
        from lightspeed_evaluation.core.script.manager import ScriptExecutionManager

        return ScriptExecutionManager
    if name == "ScriptExecutionError":
        from lightspeed_evaluation.core.system.exceptions import ScriptExecutionError

        return ScriptExecutionError

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
