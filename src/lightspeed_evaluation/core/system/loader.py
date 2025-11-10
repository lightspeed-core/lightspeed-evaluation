"""Configuration loading for evaluation framework."""

import logging
from pathlib import Path
from typing import Any, Optional

import yaml

from lightspeed_evaluation.core.models import (
    APIConfig,
    CoreConfig,
    EmbeddingConfig,
    EvaluationData,
    LLMConfig,
    LoggingConfig,
    OutputConfig,
    SystemConfig,
    VisualizationConfig,
    GEvalConfig,
)
from lightspeed_evaluation.core.system.setup import (
    setup_environment_variables,
    setup_logging,
)

# Global metric mapping sets (populated dynamically from system config)
TURN_LEVEL_METRICS: set[str] = set()
CONVERSATION_LEVEL_METRICS: set[str] = set()


def _load_geval_metrics(registry_path: str) -> None:
    """Load GEval metrics from registry file and add them to available metrics.

    Args:
        registry_path: Path to the GEval metrics registry YAML file
    """
    logger = logging.getLogger(__name__)

    # Resolve registry path
    path = Path(registry_path)
    if not path.is_absolute():
        path = Path.cwd() / path

    # Check if file exists
    if not path.exists():
        logger.warning(
            "GEval registry file not found at %s. "
            "GEval metrics will not be available for validation.",
            path,
        )
        return

    # Load registry file
    try:
        with open(path, encoding="utf-8") as f:
            registry = yaml.safe_load(f)

        if not registry:
            logger.warning("GEval registry file %s is empty", path)
            return

        if not isinstance(registry, dict):
            logger.warning(
                "GEval registry file %s has invalid format (expected dict)", path
            )
            return

        # Add GEval metrics to available metrics
        # Note: The registry may contain both turn-level and conversation-level metrics
        # We need to determine which is which based on common naming patterns or
        # add them all to both sets (metrics are validated based on context during use)

        for metric_name in registry.keys():
            # Add with geval: prefix to both turn and conversation level metrics
            # The actual validation of which level is appropriate happens during evaluation
            metric_identifier = f"geval:{metric_name}"

            # Add to both sets - the actual usage context determines applicability
            TURN_LEVEL_METRICS.add(metric_identifier)
            CONVERSATION_LEVEL_METRICS.add(metric_identifier)

        logger.debug("Loaded %d GEval metrics from %s", len(registry), path)

    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.error("Failed to load GEval registry from %s: %s", path, e)


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

    # Load GEval metrics from registry if enabled
    if system_config.geval.enabled:
        _load_geval_metrics(system_config.geval.registry_path)


def validate_metrics(
    turn_metrics: list[str], conversation_metrics: list[str]
) -> list[str]:
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
        self.evaluation_data: Optional[list[EvaluationData]] = None
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

    def _create_system_config(self, config_data: dict[str, Any]) -> SystemConfig:
        """Create SystemConfig object from validated configuration data."""
        metrics_metadata = config_data.get("metrics_metadata", {})
        return SystemConfig(
            core=CoreConfig(**config_data.get("core", {})),
            llm=LLMConfig(**config_data.get("llm", {})),
            embedding=EmbeddingConfig(**config_data.get("embedding") or {}),
            api=APIConfig(**config_data.get("api", {})),
            output=OutputConfig(**config_data.get("output", {})),
            logging=LoggingConfig(**config_data.get("logging", {})),
            visualization=VisualizationConfig(**config_data.get("visualization", {})),
            geval=GEvalConfig(**config_data.get("geval", {})),
            default_turn_metrics_metadata=metrics_metadata.get("turn_level", {}),
            default_conversation_metrics_metadata=metrics_metadata.get(
                "conversation_level", {}
            ),
        )
