"""Configuration loading for LSC Evaluation Framework."""

import logging
import os
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import yaml
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from .models import EvaluationData

# Global metric mapping sets (populated dynamically from system config)
TURN_LEVEL_METRICS: set[str] = set()
CONVERSATION_LEVEL_METRICS: set[str] = set()

# Default CSV columns
DEFAULT_CSV_COLUMNS = [
    "conversation_group_id",
    "turn_id",
    "metric_identifier",
    "result",
    "score",
    "threshold",
    "reason",
    "execution_time",
]


def setup_environment_variables(config_path: str) -> None:
    """Early setup of environment variables from system config. Called before any other imports."""
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config_data = yaml.safe_load(f)

        # Set environment variables from config
        env_config = config_data.get("environment", {})
        for env_var, value in env_config.items():
            os.environ[env_var] = str(value)

    except (FileNotFoundError, yaml.YAMLError, KeyError) as e:
        # Fallback to hardcoded defaults if config loading fails
        print(f"Warning: Could not load environment config from {config_path}: {e}")
        print("Using fallback environment settings...")
        os.environ["DEEPEVAL_TELEMETRY_OPT_OUT"] = "YES"
        os.environ["DEEPEVAL_DISABLE_PROGRESS_BAR"] = "YES"
        os.environ["LITELLM_LOG_LEVEL"] = "ERROR"


def setup_logging(logging_config: Dict[str, Any]) -> logging.Logger:
    """Configure logging for application and packages."""
    # Get logging settings with new structure
    source_level = getattr(logging, logging_config.get("source_level", "INFO").upper())
    package_level = getattr(
        logging, logging_config.get("package_level", "WARNING").upper()
    )
    log_format = logging_config.get(
        "format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Configure root logger for our application
    logging.basicConfig(
        level=source_level,
        format=log_format,
        force=True,  # Override any existing configuration
    )

    # Set logging levels for packages using system configuration
    package_overrides = logging_config.get("package_overrides", {})

    # Apply default package level to commonly problematic packages if not explicitly configured
    default_noisy_packages = [
        "urllib3",
        "requests",
        "httpx",
        "openai",
        "anthropic",
        "litellm",
        "ragas",
        "deepeval",
        "matplotlib",
        "seaborn",
    ]

    for package_name in default_noisy_packages:
        if package_name not in package_overrides:
            logging.getLogger(package_name).setLevel(package_level)

    # Apply specific package overrides from system configuration
    for package_name, override_level in package_overrides.items():
        try:
            override_level_obj = getattr(logging, override_level.upper())
            logging.getLogger(package_name).setLevel(override_level_obj)
        except AttributeError:
            print(
                f"Warning: Invalid log level '{override_level}' for package '{package_name}'"
            )

    # Get logger for our application
    logger = logging.getLogger("lsc_eval")
    logger.setLevel(source_level)

    logger.debug(
        "Logging configured: Source=%s, Packages=%s",
        logging.getLevelName(source_level),
        logging.getLevelName(package_level),
    )
    if package_overrides:
        logger.debug("Package overrides applied: %s", package_overrides)

    return logger


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


class SystemConfig(BaseModel):
    """System configuration model."""

    # LLM Configuration
    llm_provider: str = Field(default="openai")
    llm_model: str = Field(default="gpt-4o-mini")
    llm_temperature: float = Field(default=0.0)
    llm_max_tokens: int = Field(default=512)
    llm_timeout: int = Field(default=300)
    llm_retries: int = Field(default=3)

    # Logging Configuration
    logging_source_level: str = Field(default="INFO")
    logging_package_level: str = Field(default="WARNING")
    logging_format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    logging_show_timestamps: bool = Field(default=True)
    logging_package_overrides: Dict[str, str] = Field(default_factory=dict)

    # Output Configuration
    output_dir: str = Field(default="./eval_output")
    base_filename: str = Field(default="evaluation")
    csv_format: bool = Field(default=True)
    json_format: bool = Field(default=True)
    txt_format: bool = Field(default=True)
    include_graphs: bool = Field(default=True)
    csv_columns: List[str] = Field(default_factory=list)

    # Visualization Configuration
    visualization_figsize: List[int] = Field(default=[12, 8])
    visualization_dpi: int = Field(default=300)

    # Default metrics metadata from system config
    default_turn_metrics_metadata: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict
    )
    default_conversation_metrics_metadata: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict
    )


class ConfigLoader:
    """Configuration loader for LSC Evaluation Framework."""

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

        # Extract and flatten configuration
        llm_config = config_data.get("llm", {})
        output_config = config_data.get("output", {})
        visualization_config = config_data.get("visualization", {})

        self.system_config = SystemConfig(
            # LLM settings
            llm_provider=llm_config.get("provider", "openai"),
            llm_model=llm_config.get("model", "gpt-4o-mini"),
            llm_temperature=llm_config.get("temperature", 0.0),
            llm_max_tokens=llm_config.get("max_tokens", 512),
            llm_timeout=llm_config.get("timeout", 300),
            llm_retries=llm_config.get("num_retries", 3),
            # Logging settings (flattened)
            logging_source_level=logging_config.get("source_level", "INFO"),
            logging_package_level=logging_config.get("package_level", "WARNING"),
            logging_format=logging_config.get(
                "format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            ),
            logging_show_timestamps=logging_config.get("show_timestamps", True),
            logging_package_overrides=logging_config.get("package_overrides", {}),
            # Output settings
            output_dir=output_config.get("base_directory", "./eval_output"),
            base_filename=output_config.get("base_filename", "evaluation"),
            csv_format=output_config.get("formats", {}).get("csv", True),
            json_format=output_config.get("formats", {}).get("json", True),
            txt_format=output_config.get("formats", {}).get("txt", True),
            include_graphs=output_config.get("include_graphs", True),
            csv_columns=output_config.get("csv_columns", DEFAULT_CSV_COLUMNS),
            # Visualization settings
            visualization_figsize=visualization_config.get("figsize", [12, 8]),
            visualization_dpi=visualization_config.get("dpi", 300),
            # Default metrics metadata from system config
            default_turn_metrics_metadata=metrics_metadata.get("turn_level", {}),
            default_conversation_metrics_metadata=metrics_metadata.get(
                "conversation_level", {}
            ),
        )

        self.logger.debug(
            "System config loaded: %s/%s",
            self.system_config.llm_provider,
            self.system_config.llm_model,
        )
        return self.system_config

    def get_llm_config_dict(self) -> Dict[str, Any]:
        """Get system config as dictionary for LLMManager."""
        if not self.system_config:
            raise ValueError("System config not loaded")

        return {
            "llm": {
                "provider": self.system_config.llm_provider,
                "model": self.system_config.llm_model,
                "temperature": self.system_config.llm_temperature,
                "max_tokens": self.system_config.llm_max_tokens,
                "timeout": self.system_config.llm_timeout,
                "num_retries": self.system_config.llm_retries,
            }
        }
