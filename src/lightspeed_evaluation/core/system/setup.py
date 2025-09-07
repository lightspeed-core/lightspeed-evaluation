"""System setup and initialization utilities."""

import logging
import os
from typing import Any, Dict

import yaml

from ..constants import (
    DEFAULT_LOG_FORMAT,
    DEFAULT_LOG_PACKAGE_LEVEL,
    DEFAULT_LOG_SOURCE_LEVEL,
)


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
    source_level = getattr(
        logging, logging_config.get("source_level", DEFAULT_LOG_SOURCE_LEVEL).upper()
    )
    package_level = getattr(
        logging, logging_config.get("package_level", DEFAULT_LOG_PACKAGE_LEVEL).upper()
    )
    log_format = logging_config.get("format", DEFAULT_LOG_FORMAT)

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
    logger = logging.getLogger("lightspeed_evaluation")
    logger.setLevel(source_level)

    logger.debug(
        "Logging configured: Source=%s, Packages=%s",
        logging.getLevelName(source_level),
        logging.getLevelName(package_level),
    )
    if package_overrides:
        logger.debug("Package overrides applied: %s", package_overrides)

    return logger
