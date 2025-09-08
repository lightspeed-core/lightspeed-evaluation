"""LightSpeed Evaluation Framework.

Main components:
- EvaluationPipeline: Runs complete evaluation pipeline
- Runner: Simple runner for command-line usage
- Core modules organized by functionality (config, llm, metrics, output)

Usage:
    from lightspeed_evaluation import EvaluationPipeline, ConfigLoader
    loader = ConfigLoader()
    pipeline = EvaluationPipeline(loader)
    results = pipeline.run_evaluation(evaluation_data)
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
    ToolCallData,
    TurnData,
    VisualizationConfig,
)

# Output handling
from .core.output import GraphGenerator, OutputHandler

# System config
from .core.system import ConfigLoader, DataValidator, SystemConfig
from .core.system.exceptions import APIError, DataValidationError, EvaluationError

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
    "ToolCallData",
    "EvaluationResult",
    "DataValidator",
    "LLMManager",
    "APIClient",
    "OutputHandler",
    "GraphGenerator",
    "APIError",
    "DataValidationError",
    "EvaluationError",
]
