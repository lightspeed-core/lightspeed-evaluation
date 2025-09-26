"""LightSpeed Evaluation Framework.

Main components:
- EvaluationPipeline: Runs complete evaluation pipeline
- Runner: Simple runner for command-line usage
- Core modules organized by functionality (config, llm, metrics, output)
"""

__version__ = "0.1.0"

# Core components
from .core.api import APIClient
from .core.llm import LLMManager
from .core.models import (
    APIConfig,
    EvaluationData,
    EvaluationResult,
    LLMConfig,
    LoggingConfig,
    OutputConfig,
    TurnData,
    VisualizationConfig,
)

# Output handling
from .core.output import GraphGenerator, OutputHandler
from .core.script import ScriptExecutionManager

# System config
from .core.system import ConfigLoader, DataValidator, SystemConfig
from .core.system.exceptions import (
    APIError,
    DataValidationError,
    EvaluationError,
    ScriptExecutionError,
)

# Main pipeline
from .pipeline.evaluation import EvaluationPipeline

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
