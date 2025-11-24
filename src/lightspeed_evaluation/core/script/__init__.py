"""Script execution module for evaluation framework."""

from typing import TYPE_CHECKING

from lightspeed_evaluation.core.system.lazy_import import create_lazy_getattr

if TYPE_CHECKING:
    # ruff: noqa: F401
    from lightspeed_evaluation.core.script.manager import ScriptExecutionManager
    from lightspeed_evaluation.core.system.exceptions import ScriptExecutionError

_LAZY_IMPORTS = {
    "ScriptExecutionManager": (
        "lightspeed_evaluation.core.script.manager",
        "ScriptExecutionManager",
    ),
    "ScriptExecutionError": (
        "lightspeed_evaluation.core.system.exceptions",
        "ScriptExecutionError",
    ),
}

__getattr__ = create_lazy_getattr(_LAZY_IMPORTS, __name__)
__all__ = list(_LAZY_IMPORTS.keys())
