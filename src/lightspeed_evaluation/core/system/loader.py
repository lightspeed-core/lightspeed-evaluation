"""Configuration loading for evaluation framework."""

import logging
from typing import Any, Dict, List, Optional

import yaml

from ..models import (
    APIConfig,
    EvaluationData,
    LLMConfig,
    LoggingConfig,
    OutputConfig,
    SystemConfig,
    VisualizationConfig,
)
from .setup import setup_environment_variables, setup_logging

# Global metric mapping sets (populated dynamically from system config)
TURN_LEVEL_METRICS: set[str] = set()
CONVERSATION_LEVEL_METRICS: set[str] = set()


def populate_metric_mappings(system_config: "SystemConfig") -> None:
    """Populate global metric mapping sets from system config."""
    TURN_LEVEL_METRICS.clear()
    CONVERSATION_LEVEL_METRICS.clear()

    # Extract metrics metadata from validated system config
    metrics_metadata = {
        "turn_level": system_config.default_turn_metrics_metadata,
        "conversation_level": system_config.default_conversation_metrics_metadata,
    }

    # Process turn-level metrics
    turn_level = metrics_metadata.get("turn_level", {})
    for metric_name in turn_level.keys():
        TURN_LEVEL_METRICS.add(metric_name)

    # Process conversation-level metrics
    conversation_level = metrics_metadata.get("conversation_level", {})
    for metric_name in conversation_level.keys():
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


class ConfigLoader:  # pylint: disable=too-few-public-methods
    """Configuration loader for evaluation framework."""

    def __init__(self) -> None:
        """Initialize Config Loader."""
        self.system_config: Optional[SystemConfig] = None
        self.evaluation_data: Optional[List[EvaluationData]] = None
        self.logger: Optional[logging.Logger] = None

    def load_system_config(self, config_path: str) -> SystemConfig:
        """Load system configuration from YAML file."""
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config_data = yaml.safe_load(f)
        except FileNotFoundError as exc:
            raise ValueError(f"Configuration file not found: {config_path}") from exc
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML syntax in {config_path}: {e}") from e

        # Handle empty/invalid YAML safely
        if config_data is None:
            raise ValueError("Empty or invalid YAML configuration file")
        if not isinstance(config_data, dict):
            raise ValueError(
                f"YAML root must be a dictionary, got {type(config_data).__name__}"
            )

        # Load & validate system config
        self.system_config = self._create_system_config(config_data)

        # Setup environment and logging
        setup_environment_variables(config_data)
        self.logger = setup_logging(self.system_config.logging)

        # Populate metric mappings
        populate_metric_mappings(self.system_config)

        self.logger.debug("System config loaded successfully")
        return self.system_config

    def _create_system_config(self, config_data: Dict[str, Any]) -> SystemConfig:
        """Create SystemConfig object from validated configuration data."""
        metrics_metadata = config_data.get("metrics_metadata", {})
        return SystemConfig(
            llm=LLMConfig(**config_data.get("llm", {})),
            api=APIConfig(**config_data.get("api", {})),
            output=OutputConfig(**config_data.get("output", {})),
            logging=LoggingConfig(**config_data.get("logging", {})),
            visualization=VisualizationConfig(**config_data.get("visualization", {})),
            default_turn_metrics_metadata=metrics_metadata.get("turn_level", {}),
            default_conversation_metrics_metadata=metrics_metadata.get(
                "conversation_level", {}
            ),
        )
