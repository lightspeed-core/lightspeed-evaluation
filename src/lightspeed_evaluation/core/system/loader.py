"""Configuration loading for evaluation framework."""

import logging
from typing import Any, Dict, List, Optional

import yaml

from ..constants import (
    DEFAULT_API_BASE,
    DEFAULT_API_TIMEOUT,
    DEFAULT_BASE_FILENAME,
    DEFAULT_CSV_FORMAT,
    DEFAULT_ENDPOINT_TYPE,
    DEFAULT_INCLUDE_GRAPHS,
    DEFAULT_JSON_FORMAT,
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
    DEFAULT_TXT_FORMAT,
    DEFAULT_VISUALIZATION_DPI,
    DEFAULT_VISUALIZATION_FIGSIZE,
)
from ..models import (
    APIConfig,
    EvaluationData,
    LLMConfig,
    LoggingConfig,
    OutputConfig,
    SystemConfig,
    VisualizationConfig,
)
from .setup import setup_logging

# Global metric mapping sets (populated dynamically from system config)
TURN_LEVEL_METRICS: set[str] = set()
CONVERSATION_LEVEL_METRICS: set[str] = set()


def populate_metric_mappings(metrics_metadata: Dict[str, Any]) -> None:
    """Populate global metric mapping sets from system config metadata."""
    TURN_LEVEL_METRICS.clear()
    CONVERSATION_LEVEL_METRICS.clear()

    # Process turn-level metrics
    turn_level = metrics_metadata.get("turn_level", {})
    for metric_name, metadata in turn_level.items():
        if metadata.get("type") == "turn":
            TURN_LEVEL_METRICS.add(metric_name)

    # Process conversation-level metrics
    conversation_level = metrics_metadata.get("conversation_level", {})
    for metric_name, metadata in conversation_level.items():
        if metadata.get("type") == "conversation":
            CONVERSATION_LEVEL_METRICS.add(metric_name)


def validate_metrics(
    turn_metrics: List[str], conversation_metrics: List[str]
) -> List[str]:
    """Validate that provided metrics are recognized."""
    errors = []

    # Check turn-level metrics
    for metric in turn_metrics:
        if metric not in TURN_LEVEL_METRICS:
            errors.append(f"'{metric}' is not a recognized turn-level metric")

    # Check conversation-level metrics
    for metric in conversation_metrics:
        if metric not in CONVERSATION_LEVEL_METRICS:
            errors.append(f"'{metric}' is not a recognized conversation-level metric")

    return errors


class ConfigLoader:
    """Configuration loader for evaluation framework."""

    def __init__(self) -> None:
        """Initialize Config Loader."""
        self.system_config: Optional[SystemConfig] = None
        self.evaluation_data: Optional[List[EvaluationData]] = None
        self.logger: Optional[logging.Logger] = None

    def load_system_config(self, config_path: str) -> SystemConfig:
        """Load system configuration from YAML file."""
        with open(config_path, "r", encoding="utf-8") as f:
            config_data = yaml.safe_load(f)

        # Setup logging first (from config)
        logging_config = config_data.get("logging", {})
        self.logger = setup_logging(logging_config)

        # Populate metric mappings from system config
        metrics_metadata = config_data.get("metrics_metadata", {})
        populate_metric_mappings(metrics_metadata)

        # Extract configuration sections
        llm_config_data = config_data.get("llm", {})
        output_config_data = config_data.get("output", {})
        visualization_config_data = config_data.get("visualization", {})
        api_config_data = config_data.get("api", {})

        # Create individual config models
        llm_config = LLMConfig(
            provider=llm_config_data.get("provider", DEFAULT_LLM_PROVIDER),
            model=llm_config_data.get("model", DEFAULT_LLM_MODEL),
            temperature=llm_config_data.get("temperature", DEFAULT_LLM_TEMPERATURE),
            max_tokens=llm_config_data.get("max_tokens", DEFAULT_LLM_MAX_TOKENS),
            timeout=llm_config_data.get("timeout", DEFAULT_API_TIMEOUT),
            num_retries=llm_config_data.get("num_retries", DEFAULT_LLM_RETRIES),
        )

        api_config = APIConfig(
            enabled=api_config_data.get("enabled", True),
            api_base=api_config_data.get("api_base", DEFAULT_API_BASE),
            endpoint_type=api_config_data.get("endpoint_type", DEFAULT_ENDPOINT_TYPE),
            timeout=api_config_data.get("timeout", DEFAULT_API_TIMEOUT),
            provider=api_config_data.get("provider", DEFAULT_LLM_PROVIDER),
            model=api_config_data.get("model", DEFAULT_LLM_MODEL),
            no_tools=api_config_data.get("no_tools", False),
            system_prompt=api_config_data.get("system_prompt"),
        )

        output_config = OutputConfig(
            output_dir=output_config_data.get("base_directory", DEFAULT_OUTPUT_DIR),
            base_filename=output_config_data.get(
                "base_filename", DEFAULT_BASE_FILENAME
            ),
            csv_format=output_config_data.get("formats", {}).get(
                "csv", DEFAULT_CSV_FORMAT
            ),
            json_format=output_config_data.get("formats", {}).get(
                "json", DEFAULT_JSON_FORMAT
            ),
            txt_format=output_config_data.get("formats", {}).get(
                "txt", DEFAULT_TXT_FORMAT
            ),
            include_graphs=output_config_data.get(
                "include_graphs", DEFAULT_INCLUDE_GRAPHS
            ),
        )

        logging_config_obj = LoggingConfig(
            source_level=logging_config.get("source_level", DEFAULT_LOG_SOURCE_LEVEL),
            package_level=logging_config.get(
                "package_level", DEFAULT_LOG_PACKAGE_LEVEL
            ),
            log_format=logging_config.get("format", DEFAULT_LOG_FORMAT),
            show_timestamps=logging_config.get(
                "show_timestamps", DEFAULT_LOG_SHOW_TIMESTAMPS
            ),
        )

        visualization_config = VisualizationConfig(
            figsize=visualization_config_data.get(
                "figsize", DEFAULT_VISUALIZATION_FIGSIZE
            ),
            dpi=visualization_config_data.get("dpi", DEFAULT_VISUALIZATION_DPI),
        )

        self.system_config = SystemConfig(
            llm=llm_config,
            api=api_config,
            output=output_config,
            logging=logging_config_obj,
            visualization=visualization_config,
            default_turn_metrics_metadata=metrics_metadata.get("turn_level", {}),
            default_conversation_metrics_metadata=metrics_metadata.get(
                "conversation_level", {}
            ),
        )

        # Log system config loaded (avoiding pylint FieldInfo issues)
        self.logger.debug("System config loaded successfully")
        return self.system_config

    def get_llm_config_dict(self) -> Dict[str, Any]:
        """Get system config as dictionary for LLMManager."""
        if not self.system_config:
            raise ValueError(
                "System config not loaded - call load_system_config() first"
            )

        config = self.system_config

        llm = config.llm
        return {
            "llm": {
                "provider": getattr(llm, "provider", DEFAULT_LLM_PROVIDER),
                "model": getattr(llm, "model", DEFAULT_LLM_MODEL),
                "temperature": getattr(llm, "temperature", DEFAULT_LLM_TEMPERATURE),
                "max_tokens": getattr(llm, "max_tokens", DEFAULT_LLM_MAX_TOKENS),
                "timeout": getattr(llm, "timeout", DEFAULT_API_TIMEOUT),
                "num_retries": getattr(llm, "num_retries", DEFAULT_LLM_RETRIES),
            }
        }
