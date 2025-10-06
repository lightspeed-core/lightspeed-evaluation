"""System configuration, validation, and utilities."""

# Constants & Config
from lightspeed_evaluation.core.constants import (
    DEFAULT_API_BASE,
    DEFAULT_API_TIMEOUT,
    DEFAULT_BASE_FILENAME,
    DEFAULT_ENDPOINT_TYPE,
    DEFAULT_LLM_MAX_TOKENS,
    DEFAULT_LLM_MODEL,
    DEFAULT_LLM_PROVIDER,
    DEFAULT_LLM_RETRIES,
    DEFAULT_LLM_TEMPERATURE,
    DEFAULT_LOG_FORMAT,
    DEFAULT_LOG_PACKAGE_LEVEL,
    DEFAULT_LOG_SHOW_TIMESTAMPS,
    DEFAULT_LOG_SOURCE_LEVEL,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_VISUALIZATION_DPI,
    DEFAULT_VISUALIZATION_FIGSIZE,
    SUPPORTED_ENDPOINT_TYPES,
    SUPPORTED_RESULT_STATUSES,
)
from lightspeed_evaluation.core.models import SystemConfig
from lightspeed_evaluation.core.system.exceptions import (
    APIError,
    ConfigurationError,
    DataValidationError,
    EvaluationError,
    LLMError,
    MetricError,
)
from lightspeed_evaluation.core.system.loader import ConfigLoader
from lightspeed_evaluation.core.system.setup import (
    setup_environment_variables,
    setup_logging,
)
from lightspeed_evaluation.core.system.validator import DataValidator

__all__ = [
    # Configuration
    "ConfigLoader",
    "SystemConfig",
    "DataValidator",
    # Constants
    "DEFAULT_API_BASE",
    "DEFAULT_API_TIMEOUT",
    "DEFAULT_BASE_FILENAME",
    "DEFAULT_ENDPOINT_TYPE",
    "DEFAULT_LLM_MAX_TOKENS",
    "DEFAULT_LLM_MODEL",
    "DEFAULT_LLM_PROVIDER",
    "DEFAULT_LLM_RETRIES",
    "DEFAULT_LLM_TEMPERATURE",
    "DEFAULT_LOG_FORMAT",
    "DEFAULT_LOG_PACKAGE_LEVEL",
    "DEFAULT_LOG_SHOW_TIMESTAMPS",
    "DEFAULT_LOG_SOURCE_LEVEL",
    "DEFAULT_OUTPUT_DIR",
    "DEFAULT_VISUALIZATION_DPI",
    "DEFAULT_VISUALIZATION_FIGSIZE",
    "SUPPORTED_ENDPOINT_TYPES",
    "SUPPORTED_RESULT_STATUSES",
    # Exceptions
    "APIError",
    "ConfigurationError",
    "DataValidationError",
    "EvaluationError",
    "LLMError",
    "MetricError",
    # Setup functions
    "setup_environment_variables",
    "setup_logging",
]
