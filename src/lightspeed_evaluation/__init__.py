"""LightSpeed Evaluation Framework.

Main components:
- EvaluationDriver: Runs complete evaluation
- Runner: Simple runner for command-line usage
- Core modules organized by functionality (config, llm, metrics, output)

Usage:
    from lightspeed_evaluation import EvaluationDriver, ConfigLoader
    loader = ConfigLoader()
    driver = EvaluationDriver(loader)
    results = driver.run_evaluation(evaluation_data)
"""

__version__ = "0.1.0"

# Main driver
from .drivers.evaluation import EvaluationDriver

# Core components
from .core.llm import LLMManager
from .core.config import (
    ConfigLoader,
    SystemConfig,
    EvaluationData,
    TurnData,
    EvaluationResult,
    DataValidator,
)
from .core.config import LLMConfig
from .core.metrics import RagasMetrics, DeepEvalMetrics, CustomMetrics

# Output handling
from .core.output import OutputHandler, GraphGenerator

__all__ = [
    # Main components
    "EvaluationDriver",
    "LLMManager",
    "ConfigLoader",
    "SystemConfig",
    "EvaluationData",
    "TurnData",
    "EvaluationResult",
    "DataValidator",
    "LLMConfig",
    # Metrics
    "RagasMetrics",
    "DeepEvalMetrics",
    "CustomMetrics",
    # Output handling
    "OutputHandler",
    "GraphGenerator",
]
