"""LightSpeed Evaluation Framework.

Main components:
- EvaluationPipeline: Runs complete evaluation pipeline
- Runner: Simple runner for command-line usage
- Core modules organized by functionality (config, llm, metrics, output)
"""

from typing import TYPE_CHECKING

from lightspeed_evaluation.core.system.lazy_import import create_lazy_getattr

if TYPE_CHECKING:
    # ruff: noqa: F401
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

__version__ = "0.3.0"

_LAZY_IMPORTS = {
    # Main pipeline
    "EvaluationPipeline": (
        "lightspeed_evaluation.pipeline.evaluation",
        "EvaluationPipeline",
    ),
    # System config
    "ConfigLoader": ("lightspeed_evaluation.core.system", "ConfigLoader"),
    "SystemConfig": ("lightspeed_evaluation.core.system", "SystemConfig"),
    "DataValidator": ("lightspeed_evaluation.core.system", "DataValidator"),
    # Models
    "LLMConfig": ("lightspeed_evaluation.core.models", "LLMConfig"),
    "APIConfig": ("lightspeed_evaluation.core.models", "APIConfig"),
    "OutputConfig": ("lightspeed_evaluation.core.models", "OutputConfig"),
    "LoggingConfig": ("lightspeed_evaluation.core.models", "LoggingConfig"),
    "VisualizationConfig": ("lightspeed_evaluation.core.models", "VisualizationConfig"),
    "EvaluationData": ("lightspeed_evaluation.core.models", "EvaluationData"),
    "TurnData": ("lightspeed_evaluation.core.models", "TurnData"),
    "EvaluationResult": ("lightspeed_evaluation.core.models", "EvaluationResult"),
    # Core components
    "LLMManager": ("lightspeed_evaluation.core.llm", "LLMManager"),
    "APIClient": ("lightspeed_evaluation.core.api", "APIClient"),
    # Output handling
    "OutputHandler": ("lightspeed_evaluation.core.output", "OutputHandler"),
    "GraphGenerator": ("lightspeed_evaluation.core.output", "GraphGenerator"),
    # Script execution
    "ScriptExecutionManager": (
        "lightspeed_evaluation.core.script",
        "ScriptExecutionManager",
    ),
    # Exceptions
    "ScriptExecutionError": (
        "lightspeed_evaluation.core.system.exceptions",
        "ScriptExecutionError",
    ),
    "APIError": ("lightspeed_evaluation.core.system.exceptions", "APIError"),
    "DataValidationError": (
        "lightspeed_evaluation.core.system.exceptions",
        "DataValidationError",
    ),
    "EvaluationError": (
        "lightspeed_evaluation.core.system.exceptions",
        "EvaluationError",
    ),
}

__getattr__ = create_lazy_getattr(_LAZY_IMPORTS, __name__)
__all__ = list(_LAZY_IMPORTS.keys())
