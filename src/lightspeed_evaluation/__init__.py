"""LightSpeed Evaluation Framework.

Main components:
- EvaluationPipeline: Runs complete evaluation pipeline
- Runner: Simple runner for command-line usage
- Core modules organized by functionality (config, llm, metrics, output)

This module uses lazy imports to improve initial load time.
Heavy modules like metrics frameworks (ragas, deepeval) and visualization
libraries (matplotlib, seaborn) are only imported when actually used.
"""

from typing import TYPE_CHECKING, Any

__version__ = "0.2.0"

# Type checking imports - only evaluated by type checkers, not at runtime
if TYPE_CHECKING:
    from lightspeed_evaluation.core.api import APIClient
    from lightspeed_evaluation.core.llm import LLMManager
    from lightspeed_evaluation.core.models import (
        APIConfig,
        EvaluationData,
        EvaluationResult,
        LLMConfig,
        LoggingConfig,
        OutputConfig,
        TurnData,
        VisualizationConfig,
    )
    from lightspeed_evaluation.core.output import GraphGenerator, OutputHandler
    from lightspeed_evaluation.core.script import ScriptExecutionManager
    from lightspeed_evaluation.core.system import (
        ConfigLoader,
        DataValidator,
        SystemConfig,
    )
    from lightspeed_evaluation.core.system.exceptions import (
        APIError,
        DataValidationError,
        EvaluationError,
        ScriptExecutionError,
    )
    from lightspeed_evaluation.pipeline.evaluation import EvaluationPipeline

# Only define __all__ for documentation purposes
# Actual imports are lazy - modules are imported only when accessed
__all__ = [
    "EvaluationPipeline",
    "ConfigLoader",
    "SystemConfig",
    "LLMConfig",
    "APIConfig",
    "OutputConfig",
    "LoggingConfig",
    "VisualizationConfig",
    "EvaluationData",
    "TurnData",
    "EvaluationResult",
    "DataValidator",
    "LLMManager",
    "APIClient",
    "OutputHandler",
    "GraphGenerator",
    "ScriptExecutionManager",
    "ScriptExecutionError",
    "APIError",
    "DataValidationError",
    "EvaluationError",
]


def __getattr__(name: str) -> Any:  # pylint: disable=too-many-return-statements
    """Lazy import mechanism for package-level imports.

    This allows 'from lightspeed_evaluation import X' to work while
    deferring actual imports until the attribute is accessed.
    """
    # pylint: disable=import-outside-toplevel
    # Map of attribute names to their module paths
    _import_map = {
        # Pipeline
        "EvaluationPipeline": "lightspeed_evaluation.pipeline.evaluation",
        # System config
        "ConfigLoader": "lightspeed_evaluation.core.system",
        "SystemConfig": "lightspeed_evaluation.core.system",
        "DataValidator": "lightspeed_evaluation.core.system",
        # Models
        "LLMConfig": "lightspeed_evaluation.core.models",
        "APIConfig": "lightspeed_evaluation.core.models",
        "OutputConfig": "lightspeed_evaluation.core.models",
        "LoggingConfig": "lightspeed_evaluation.core.models",
        "VisualizationConfig": "lightspeed_evaluation.core.models",
        "EvaluationData": "lightspeed_evaluation.core.models",
        "TurnData": "lightspeed_evaluation.core.models",
        "EvaluationResult": "lightspeed_evaluation.core.models",
        # LLM
        "LLMManager": "lightspeed_evaluation.core.llm",
        # API
        "APIClient": "lightspeed_evaluation.core.api",
        # Output
        "OutputHandler": "lightspeed_evaluation.core.output",
        "GraphGenerator": "lightspeed_evaluation.core.output",
        # Script
        "ScriptExecutionManager": "lightspeed_evaluation.core.script",
        # Exceptions
        "ScriptExecutionError": "lightspeed_evaluation.core.system.exceptions",
        "APIError": "lightspeed_evaluation.core.system.exceptions",
        "DataValidationError": "lightspeed_evaluation.core.system.exceptions",
        "EvaluationError": "lightspeed_evaluation.core.system.exceptions",
    }

    if name in _import_map:
        import importlib

        module = importlib.import_module(_import_map[name])
        return getattr(module, name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
