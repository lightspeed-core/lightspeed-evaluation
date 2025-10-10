"""LightSpeed Evaluation Framework.

Main components:
- EvaluationPipeline: Runs complete evaluation pipeline
- Runner: Simple runner for command-line usage
- Core modules organized by functionality (config, llm, metrics, output)
"""

__version__ = "0.2.0"

# Core components
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

# Output handling
from lightspeed_evaluation.core.output import GraphGenerator, OutputHandler
from lightspeed_evaluation.core.script import ScriptExecutionManager

# System config
from lightspeed_evaluation.core.system import ConfigLoader, DataValidator, SystemConfig
from lightspeed_evaluation.core.system.exceptions import (
    APIError,
    DataValidationError,
    EvaluationError,
    ScriptExecutionError,
)

# Main pipeline
from lightspeed_evaluation.pipeline.evaluation import EvaluationPipeline

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
