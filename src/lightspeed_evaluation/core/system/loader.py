"""Configuration loading for evaluation framework."""

import logging
from typing import Any, Optional

import yaml

from lightspeed_evaluation.core.models import (
    APIConfig,
    CoreConfig,
    EmbeddingConfig,
    EvaluationData,
    LLMConfig,
    LoggingConfig,
    SystemConfig,
    VisualizationConfig,
)
from lightspeed_evaluation.core.models.system import (
    JudgePanelConfig,
    LLMPoolConfig,
    QualityScoreConfig,
)
from lightspeed_evaluation.core.storage.config import (
    DatabaseBackendConfig,
    FileBackendConfig,
    StorageBackendConfig,
)
from lightspeed_evaluation.core.system.exceptions import ConfigurationError
from lightspeed_evaluation.core.system.setup import (
    setup_environment_variables,
    setup_logging,
)

logger = logging.getLogger(__name__)

# Supported storage backend types
SUPPORTED_STORAGE_TYPES: tuple[str, ...] = ("file", "sqlite", "postgres", "mysql")
DATABASE_STORAGE_TYPES: tuple[str, ...] = ("sqlite", "postgres", "mysql")


class ConfigLoader:  # pylint: disable=too-few-public-methods
    """Configuration loader for evaluation framework."""

    def __init__(self) -> None:
        """Initialize Config Loader."""
        self.system_config: Optional[SystemConfig] = None
        self.evaluation_data: Optional[list[EvaluationData]] = None
        self.logger: Optional[logging.Logger] = None

    @classmethod
    def from_config(cls, system_config: SystemConfig) -> "ConfigLoader":
        """Create a fully-initialized ConfigLoader from an existing SystemConfig.

        This allows programmatic use of the evaluation pipeline without
        loading configuration from a YAML file.

        Args:
            system_config: A pre-built SystemConfig instance.

        Returns:
            A fully-initialized ConfigLoader ready for pipeline use.
        """
        loader = cls()
        loader.system_config = system_config

        config_data = cls._build_config_data_from_system_config(system_config)
        setup_environment_variables(config_data)
        loader.logger = setup_logging(system_config.logging)

        return loader

    @staticmethod
    def _build_config_data_from_system_config(
        system_config: SystemConfig,
    ) -> dict[str, Any]:
        """Build the minimal config dict needed by setup_environment_variables.

        Extracts SSL-related fields so that ``create_ssl_certifi_bundle``
        can discover custom certificate paths.

        Args:
            system_config: The SystemConfig to extract SSL fields from.

        Returns:
            A dict suitable for ``setup_environment_variables``.
        """
        return {
            "llm": {
                "ssl_verify": system_config.llm.ssl_verify,
                "ssl_cert_file": system_config.llm.ssl_cert_file,
            },
        }

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

        self.logger.debug("System config loaded successfully")
        return self.system_config

    def _create_system_config(self, config_data: dict[str, Any]) -> SystemConfig:
        """Create SystemConfig object from validated configuration data."""
        metrics_metadata = config_data.get("metrics_metadata", {})
        quality_score_data = config_data.get("quality_score")

        # Process quality_score defaults before creating SystemConfig
        turn_level_metadata = metrics_metadata.get("turn_level", {})
        conversation_level_metadata = metrics_metadata.get("conversation_level", {})

        if quality_score_data:
            self._process_quality_score_defaults(
                quality_score_data,
                turn_level_metadata,
                conversation_level_metadata,
            )

        # Parse quality_score config if present
        quality_score_config = (
            QualityScoreConfig(**quality_score_data) if quality_score_data else None
        )

        # Parse llm_pool and judge_panel if present (Optional sections)
        llm_pool_data = config_data.get("llm_pool")
        llm_pool = LLMPoolConfig(**llm_pool_data) if llm_pool_data else None

        judge_panel_data = config_data.get("judge_panel")
        judge_panel = JudgePanelConfig(**judge_panel_data) if judge_panel_data else None

        # Parse storage backends with backward compatibility
        storage_data = self._get_storage_config_with_backward_compat(config_data)
        storage_backends = self._parse_storage_config(storage_data)

        return SystemConfig(
            core=CoreConfig(**config_data.get("core", {})),
            llm=LLMConfig(**config_data.get("llm", {})),
            embedding=EmbeddingConfig(**config_data.get("embedding") or {}),
            api=APIConfig(**config_data.get("api", {})),
            storage=storage_backends,
            logging=LoggingConfig(**config_data.get("logging", {})),
            visualization=VisualizationConfig(**config_data.get("visualization", {})),
            llm_pool=llm_pool,
            judge_panel=judge_panel,
            quality_score=quality_score_config,
            default_turn_metrics_metadata=turn_level_metadata,
            default_conversation_metrics_metadata=conversation_level_metadata,
        )

    def _process_quality_score_defaults(
        self,
        quality_score_config: dict[str, Any],
        turn_level_metadata: dict[str, dict[str, Any]],
        conversation_level_metadata: dict[str, dict[str, Any]],
    ) -> None:
        """Process quality_score.default to set defaults for quality score metrics.

        If quality_score.default is true, sets default: true for all metrics
        listed in quality_score.metrics. Raises error if a metric is not defined
        in turn_level or conversation_level metadata.

        Args:
            quality_score_config: The quality_score configuration dict
            turn_level_metadata: Turn-level metrics metadata (modified in-place)
            conversation_level_metadata: Conversation-level metrics metadata (modified in-place)

        Raises:
            ConfigurationError: If quality_score.default is true but a metric is not
                defined in metadata, or if metrics list is empty/missing.
        """
        if not quality_score_config:
            return

        # Check if default flag is set to true
        default_flag = quality_score_config.get("default", False)
        if not default_flag:
            return

        # Get the list of metrics for quality score
        quality_score_metrics = quality_score_config.get("metrics", [])
        if not quality_score_metrics:
            raise ConfigurationError(
                "quality_score.default is true but quality_score.metrics is empty or missing. "
                "Please specify at least one metric in quality_score.metrics."
            )

        # Process each metric
        for metric_id in quality_score_metrics:
            # Check if metric exists in turn_level or conversation_level
            if metric_id in turn_level_metadata:
                # Set default: true for this metric
                if not isinstance(turn_level_metadata[metric_id], dict):
                    turn_level_metadata[metric_id] = {}
                turn_level_metadata[metric_id]["default"] = True
            elif metric_id in conversation_level_metadata:
                # Set default: true for this metric
                if not isinstance(conversation_level_metadata[metric_id], dict):
                    conversation_level_metadata[metric_id] = {}
                conversation_level_metadata[metric_id]["default"] = True
            else:
                # Metric not found - raise error
                raise ConfigurationError(
                    f"Metric '{metric_id}' is listed in quality_score.metrics but not defined "
                    f"in metrics_metadata.turn_level or metrics_metadata.conversation_level. "
                    f"Please add metadata configuration for this metric before using it in quality_score."
                )

    def _get_storage_config_with_backward_compat(
        self, config_data: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Get storage config with backward compatibility for legacy 'output' section.

        Args:
            config_data: Raw configuration data from YAML.

        Returns:
            List of storage backend configurations.
        """
        # New format: use 'storage' section directly
        if "storage" in config_data:
            return config_data.get("storage", [])

        # Legacy format: convert 'output' section to 'storage' format
        if "output" in config_data:
            logger.warning(
                "DEPRECATION: 'output' configuration is deprecated. "
                "Please migrate to 'storage' format. "
                "See docs/configuration.md for the new format."
            )
            output_config = config_data["output"]
            # Convert legacy output config to file backend format
            file_backend = {"type": "file", **output_config}
            return [file_backend]

        # No storage config - return empty list (defaults will be used)
        return []

    def _parse_storage_config(
        self, storage_data: list[dict[str, Any]]
    ) -> list[StorageBackendConfig]:
        """Parse storage configuration into typed backend configs.

        Raises:
            ConfigurationError: If a storage entry has an unsupported ``type``.
        """
        backends: list[StorageBackendConfig] = []

        for item in storage_data:
            backend_type = item.get("type")
            if backend_type == "file":
                backends.append(FileBackendConfig(**item))
            elif backend_type in DATABASE_STORAGE_TYPES:
                backends.append(DatabaseBackendConfig(**item))
            else:
                raise ConfigurationError(
                    f"Unknown storage backend type {backend_type!r}. "
                    f"Supported types: {', '.join(SUPPORTED_STORAGE_TYPES)}"
                )

        return backends
