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
    OutputConfig,
    SystemConfig,
    VisualizationConfig,
)
from lightspeed_evaluation.core.models.system import (
    JudgePanelConfig,
    LLMPoolConfig,
)
from lightspeed_evaluation.core.system.setup import (
    setup_environment_variables,
    setup_logging,
)


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

        # Parse llm_pool and judge_panel if present (Optional sections)
        llm_pool_data = config_data.get("llm_pool")
        llm_pool = LLMPoolConfig(**llm_pool_data) if llm_pool_data else None

        judge_panel_data = config_data.get("judge_panel")
        judge_panel = JudgePanelConfig(**judge_panel_data) if judge_panel_data else None

        return SystemConfig(
            core=CoreConfig(**config_data.get("core", {})),
            llm=LLMConfig(**config_data.get("llm", {})),
            embedding=EmbeddingConfig(**config_data.get("embedding") or {}),
            api=APIConfig(**config_data.get("api", {})),
            output=OutputConfig(**config_data.get("output", {})),
            logging=LoggingConfig(**config_data.get("logging", {})),
            visualization=VisualizationConfig(**config_data.get("visualization", {})),
            llm_pool=llm_pool,
            judge_panel=judge_panel,
            default_turn_metrics_metadata=metrics_metadata.get("turn_level", {}),
            default_conversation_metrics_metadata=metrics_metadata.get(
                "conversation_level", {}
            ),
        )
